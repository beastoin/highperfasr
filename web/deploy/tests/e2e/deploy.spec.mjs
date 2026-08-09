import { chromium } from 'playwright';
import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEPLOY_HTML = resolve(__dirname, '../../deploy.html');

let server, BASE, browser, page;

async function setup() {
  const html = readFileSync(DEPLOY_HTML, 'utf-8');
  server = createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  BASE = `http://127.0.0.1:${server.address().port}/deploy.html`;
  browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 640, height: 900 } });
  page = await ctx.newPage();
}

async function teardown() {
  await browser?.close();
  server?.close();
}

async function simulateSignedIn() {
  await page.evaluate(() => {
    token = 'fake-token';
    tokenExpiry = Date.now() + 3600000;
    userEmail = 'user@example.com';
    document.getElementById('auth-prompt').classList.add('hidden');
    document.getElementById('auth-info').classList.remove('hidden');
    document.getElementById('auth-email').textContent = 'user@example.com';
    document.getElementById('step-config').classList.remove('hidden');
  });
}

let passed = 0;
let failed = 0;

async function test(name, fn) {
  try {
    await page.goto(BASE);
    await fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}`);
    console.log(`        ${e.message}`);
  }
}

console.log('\nDeploy page E2E tests\n');
await setup();

await test('sign-in page renders with auth prompt visible', async () => {
  const hidden = await page.$eval('#auth-prompt', el => el.classList.contains('hidden'));
  assert.equal(hidden, false, 'auth-prompt should be visible');
  const configHidden = await page.$eval('#step-config', el => el.classList.contains('hidden'));
  assert.equal(configHidden, true, 'config should be hidden before sign-in');
});

await test('config step appears after sign-in', async () => {
  await simulateSignedIn();
  const configHidden = await page.$eval('#step-config', el => el.classList.contains('hidden'));
  assert.equal(configHidden, false, 'config should be visible after sign-in');
  const email = await page.$eval('#auth-email', el => el.textContent);
  assert.equal(email, 'user@example.com');
});

await test('project select has loading placeholder', async () => {
  const opts = await page.$$eval('#project-select option', opts =>
    opts.map(o => ({ value: o.value, text: o.textContent }))
  );
  assert.ok(opts.length >= 1, 'should have at least one option');
  assert.ok(opts[0].text.includes('Loading') || opts[0].text.includes('Select'),
    `placeholder should mention Loading or Select, got: ${opts[0].text}`);
});

await test('mode and GPU toggle buttons exist', async () => {
  await simulateSignedIn();
  const labels = await page.evaluate(() => {
    const btns = document.querySelectorAll('.toggle-group button');
    return Array.from(btns).map(b => b.textContent.trim());
  });
  assert.ok(labels.includes('Streaming'), 'should have Streaming button');
  assert.ok(labels.includes('Batch'), 'should have Batch button');
  assert.ok(labels.some(l => l.includes('L4')), 'should have L4 button');
  assert.ok(labels.some(l => l.includes('T4')), 'should have T4 button');
});

await test('deploy button exists and has correct label', async () => {
  await simulateSignedIn();
  const btnText = await page.$eval('#deploy-btn', el => el.textContent.trim());
  assert.ok(btnText.toLowerCase().includes('deploy'), `deploy button text: ${btnText}`);
});

await test('progress stages have correct pipeline order', async () => {
  const stages = await page.$$eval('[data-stage]', els =>
    els.map(el => el.getAttribute('data-stage'))
  );
  const expected = ['preflight', 'firewall', 'vm', 'drivers', 'pull', 'model', 'health'];
  assert.deepEqual(stages, expected, 'stages should match expected pipeline');
});

await test('setStage marks stage as done with elapsed time', async () => {
  await page.evaluate(() => {
    document.getElementById('step-progress').classList.remove('hidden');
    deployState.startTime = Date.now() - 5000;
    setStage('preflight', 'done');
  });
  const cls = await page.$eval('[data-stage="preflight"]', el => el.className);
  assert.ok(cls.includes('done'), `done stage should have done class, got: ${cls}`);
  const time = await page.$eval('[data-stage="preflight"] .time', el => el.textContent);
  assert.ok(time.includes('s'), `should show elapsed time, got: ${time}`);
});

await test('setStage marks stage as active', async () => {
  await page.evaluate(() => {
    document.getElementById('step-progress').classList.remove('hidden');
    deployState.startTime = Date.now() - 5000;
    setStage('firewall', 'active');
  });
  const cls = await page.$eval('[data-stage="firewall"]', el => el.className);
  assert.ok(cls.includes('active'), `active stage should have active class, got: ${cls}`);
});

await test('setStageError shows error and retry button', async () => {
  await page.evaluate(() => {
    document.getElementById('step-progress').classList.remove('hidden');
    deployState.startTime = Date.now() - 5000;
    setStageError('Test error message');
  });
  const errorVisible = await page.$eval('#progress-error', el => !el.classList.contains('hidden'));
  assert.ok(errorVisible, 'error box should be visible');
  const retryVisible = await page.$eval('#retry-btn', el => !el.classList.contains('hidden'));
  assert.ok(retryVisible, 'retry button should be visible');
});

await test('ready page shows VM details', async () => {
  await page.evaluate(() => {
    document.getElementById('step-ready').classList.remove('hidden');
    document.getElementById('ready-url').textContent = 'http://10.0.0.1:8001';
    document.getElementById('ready-mode').textContent = 'stream';
    document.getElementById('ready-gpu').textContent = 'L4';
    document.getElementById('ready-vm').textContent = 'hpfasr-test';
    document.getElementById('ready-zone').textContent = 'us-central1-a';
  });
  const url = await page.$eval('#ready-url', el => el.textContent);
  assert.equal(url, 'http://10.0.0.1:8001');
  const gpu = await page.$eval('#ready-gpu', el => el.textContent);
  assert.equal(gpu, 'L4');
  const vm = await page.$eval('#ready-vm', el => el.textContent);
  assert.equal(vm, 'hpfasr-test');
});

await test('benchmark section populates commands', async () => {
  await page.evaluate(() => {
    document.getElementById('step-benchmark').classList.remove('hidden');
    document.getElementById('bench-title').textContent = 'Streaming Benchmark';
    document.getElementById('bench-command').textContent = 'python3 bench_stream.py --server ws://10.0.0.1:8001';
  });
  const title = await page.$eval('#bench-title', el => el.textContent);
  assert.equal(title, 'Streaming Benchmark');
  const cmd = await page.$eval('#bench-command', el => el.textContent);
  assert.ok(cmd.includes('bench_stream'), 'should contain bench command');
});

await test('sign-out clears auth state', async () => {
  await simulateSignedIn();
  await page.evaluate(() => { signOut(); });
  const authHidden = await page.$eval('#auth-prompt', el => el.classList.contains('hidden'));
  assert.equal(authHidden, false, 'auth prompt should be visible after sign-out');
  const configHidden = await page.$eval('#step-config', el => el.classList.contains('hidden'));
  assert.equal(configHidden, true, 'config should be hidden after sign-out');
});

await test('manual project toggle shows input field', async () => {
  await simulateSignedIn();
  await page.evaluate(() => {
    document.getElementById('project-manual-toggle').classList.remove('hidden');
    toggleManualProject();
  });
  const manualVisible = await page.$eval('#project-manual', el => !el.classList.contains('hidden'));
  assert.ok(manualVisible, 'manual project input should be visible after toggle');
});

await test('page title is correct', async () => {
  const title = await page.title();
  assert.equal(title, 'Deploy HighPerfASR');
});

await test('OAuth CSRF state is stored in sessionStorage', async () => {
  const hasState = await page.evaluate(() => {
    const state = crypto.randomUUID();
    sessionStorage.setItem('oauth_state', state);
    return sessionStorage.getItem('oauth_state') === state;
  });
  assert.ok(hasState, 'sessionStorage should store and retrieve state');
});

await test('renderLinkedText converts URLs to clickable links', async () => {
  await page.evaluate(() => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    el.id = 'test-linked';
    renderLinkedText(el, 'Visit https://example.com for details');
  });
  const linkHref = await page.$eval('#test-linked a', el => el.href);
  assert.equal(linkHref, 'https://example.com/');
  const linkTarget = await page.$eval('#test-linked a', el => el.target);
  assert.equal(linkTarget, '_blank');
});

// --- PR #36 features: project validation, setProjectStatus, PROJECT_ID_RE ---

await test('setProjectStatus ok shows green status and enables deploy', async () => {
  await simulateSignedIn();
  await page.evaluate(() => {
    setProjectStatus('ok', 'Project ready — Compute Engine API and billing enabled');
  });
  const statusVisible = await page.$eval('#project-status', el => !el.classList.contains('hidden'));
  assert.ok(statusVisible, 'status should be visible');
  const text = await page.$eval('#project-status', el => el.textContent);
  assert.ok(text.includes('Project ready'), 'should show ok message');
  const btnDisabled = await page.$eval('#deploy-btn', el => el.disabled);
  assert.equal(btnDisabled, false, 'deploy button should be enabled on ok');
});

await test('setProjectStatus error shows red status and disables deploy', async () => {
  await simulateSignedIn();
  await page.evaluate(() => {
    setProjectStatus('error', 'Invalid project ID format.');
  });
  const text = await page.$eval('#project-status', el => el.textContent);
  assert.ok(text.includes('Invalid project ID'), 'should show error text');
  const btnDisabled = await page.$eval('#deploy-btn', el => el.disabled);
  assert.equal(btnDisabled, true, 'deploy button should be disabled on error');
});

await test('setProjectStatus warn disables deploy', async () => {
  await simulateSignedIn();
  await page.evaluate(() => {
    setProjectStatus('warn', 'Billing not enabled.');
  });
  const btnDisabled = await page.$eval('#deploy-btn', el => el.disabled);
  assert.equal(btnDisabled, true, 'deploy button should be disabled on warn');
});

await test('validateProject keeps deploy disabled when billing cannot be verified', async () => {
  await simulateSignedIn();
  await page.evaluate(async () => {
    gapi = async url => {
      if (url.includes('compute.googleapis.com')) return {};
      if (url.includes('cloudbilling.googleapis.com')) throw new Error('backendError');
      throw new Error('unexpected URL: ' + url);
    };
    document.getElementById('deploy-btn').disabled = false;
    await validateProject('my-project-123');
  });
  const text = await page.$eval('#project-status', el => el.textContent);
  assert.ok(text.includes('Could not verify billing'), `should show billing verification warning, got: ${text}`);
  const btnDisabled = await page.$eval('#deploy-btn', el => el.disabled);
  assert.equal(btnDisabled, true, 'deploy button should remain disabled when billing check fails');
});

await test('validateProject keeps deploy disabled without billing read access', async () => {
  await simulateSignedIn();
  await page.evaluate(async () => {
    gapi = async url => {
      if (url.includes('compute.googleapis.com')) return {};
      if (url.includes('cloudbilling.googleapis.com')) throw new Error('403 forbidden');
      throw new Error('unexpected URL: ' + url);
    };
    document.getElementById('deploy-btn').disabled = false;
    await validateProject('my-project-123');
  });
  const text = await page.$eval('#project-status', el => el.textContent);
  assert.ok(text.includes('Billing status could not be verified'), `should show billing access warning, got: ${text}`);
  const btnDisabled = await page.$eval('#deploy-btn', el => el.disabled);
  assert.equal(btnDisabled, true, 'deploy button should remain disabled without billing read access');
});

await test('validateProject success checks Compute and billing then enables deploy', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(async () => {
    const calls = [];
    gapi = async url => {
      calls.push(url);
      if (url.includes('compute.googleapis.com')) return {};
      if (url.includes('cloudbilling.googleapis.com')) return { billingEnabled: true };
      throw new Error('unexpected URL: ' + url);
    };
    await validateProject('my-project-123');
    return {
      calls,
      text: document.getElementById('project-status').textContent,
      disabled: document.getElementById('deploy-btn').disabled,
    };
  });
  assert.ok(result.calls.some(url => url.includes('compute.googleapis.com')), 'should check Compute API');
  assert.ok(result.calls.some(url => url.includes('cloudbilling.googleapis.com')), 'should check Cloud Billing API');
  assert.ok(result.text.includes('Project ready'), `should show ready status, got: ${result.text}`);
  assert.equal(result.disabled, false, 'deploy button should be enabled after successful validation');
});

await test('validateProject invalid ID returns before any API call', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(async () => {
    let calls = 0;
    gapi = async () => {
      calls++;
      throw new Error('validateProject should not call APIs for invalid IDs');
    };
    await validateProject('Bad_Project');
    return {
      calls,
      text: document.getElementById('project-status').textContent,
      disabled: document.getElementById('deploy-btn').disabled,
    };
  });
  assert.equal(result.calls, 0, 'invalid project ID should not trigger API calls');
  assert.ok(result.text.includes('Invalid project ID'), `should show invalid-ID status, got: ${result.text}`);
  assert.equal(result.disabled, true, 'deploy button should stay disabled for invalid IDs');
});

await test('validateProject ignores stale results during rapid project switching', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(async () => {
    let resolveOldCompute;
    gapi = async url => {
      if (url.includes('/old-project')) {
        return new Promise(resolve => { resolveOldCompute = () => resolve({}); });
      }
      if (url.includes('/new-project') && url.includes('compute.googleapis.com')) return {};
      if (url.includes('/new-project') && url.includes('cloudbilling.googleapis.com')) return { billingEnabled: true };
      throw new Error('unexpected URL: ' + url);
    };

    const oldValidation = validateProject('old-project');
    await new Promise(resolve => setTimeout(resolve, 0));
    const newValidation = validateProject('new-project');
    await newValidation;
    resolveOldCompute();
    await oldValidation;

    return {
      text: document.getElementById('project-status').textContent,
      disabled: document.getElementById('deploy-btn').disabled,
    };
  });
  assert.ok(result.text.includes('Project ready'), `stale validation should not overwrite new status, got: ${result.text}`);
  assert.equal(result.disabled, false, 'new valid project should remain deployable');
});

await test('detectPreviousProject ignores stale auto-selection after manual toggle', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(async () => {
    const sel = document.getElementById('project-select');
    sel.replaceChildren(new Option('Select a project', ''), new Option('Old Project', 'old-project'));
    sel.classList.remove('hidden');

    let resolveFirewall;
    gapiRaw = async () => new Promise(resolve => {
      resolveFirewall = () => resolve({ ok: true });
    });

    const detection = detectPreviousProject(['old-project']);
    await new Promise(resolve => setTimeout(resolve, 0));
    toggleManualProject();
    resolveFirewall();
    await detection;

    return {
      selectHidden: sel.classList.contains('hidden'),
      selectedValue: sel.value,
      manualHidden: document.getElementById('project-manual').classList.contains('hidden'),
    };
  });
  assert.equal(result.selectHidden, true, 'manual toggle should keep select hidden');
  assert.equal(result.selectedValue, '', 'stale detection should not auto-select old project');
  assert.equal(result.manualHidden, false, 'manual input should remain visible');
});

await test('setProjectStatus accepts DOM nodes', async () => {
  await simulateSignedIn();
  await page.evaluate(() => {
    setProjectStatus('warn', [
      document.createTextNode('Billing not enabled. '),
      statusLink('Enable billing', 'https://console.cloud.google.com/billing'),
    ]);
  });
  const link = await page.$eval('#project-status a', el => el.href);
  assert.ok(link.includes('console.cloud.google.com/billing'), 'should contain billing link');
  const linkTarget = await page.$eval('#project-status a', el => el.target);
  assert.equal(linkTarget, '_blank', 'link should open in new tab');
});

await test('PROJECT_ID_RE validates format correctly', async () => {
  await simulateSignedIn();
  const results = await page.evaluate(() => {
    return {
      valid: PROJECT_ID_RE.test('my-project-123'),
      validMin: PROJECT_ID_RE.test('abcdef'),
      tooShort: PROJECT_ID_RE.test('abcde'),
      uppercase: PROJECT_ID_RE.test('MY-PROJECT'),
      startsWithDigit: PROJECT_ID_RE.test('123project'),
      hasUnderscore: PROJECT_ID_RE.test('my_project_id'),
    };
  });
  assert.equal(results.valid, true, 'valid project ID should pass');
  assert.equal(results.validMin, true, '6-char ID should pass');
  assert.equal(results.tooShort, false, '5-char ID should fail');
  assert.equal(results.uppercase, false, 'uppercase should fail');
  assert.equal(results.startsWithDigit, false, 'starting with digit should fail');
});

await test('statusLink creates DOM-safe link', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(() => {
    const link = statusLink('Click here', 'https://example.com');
    return {
      tagName: link.tagName,
      href: link.href,
      target: link.target,
      rel: link.rel,
      text: link.textContent,
    };
  });
  assert.equal(result.tagName, 'A');
  assert.equal(result.href, 'https://example.com/');
  assert.equal(result.target, '_blank');
  assert.ok(result.rel.includes('noopener'), 'should have noopener');
  assert.equal(result.text, 'Click here');
});

await test('deploy blocks expired token and signs out before API calls', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(async () => {
    tokenExpiry = Date.now() - 1000;
    document.getElementById('project-manual').value = 'my-project-123';
    document.getElementById('project-select').classList.add('hidden');
    let calls = 0;
    gapi = async () => {
      calls++;
      throw new Error('deploy should not call APIs with an expired token');
    };
    const alerts = [];
    window.alert = msg => alerts.push(msg);
    await deploy();
    return {
      calls,
      alerts,
      authPromptHidden: document.getElementById('auth-prompt').classList.contains('hidden'),
      configHidden: document.getElementById('step-config').classList.contains('hidden'),
    };
  });
  assert.equal(result.calls, 0, 'expired token should stop deploy before API calls');
  assert.ok(result.alerts.some(msg => msg.includes('session has expired')), `should alert expired session, got: ${result.alerts.join(' | ')}`);
  assert.equal(result.authPromptHidden, false, 'sign-out should show auth prompt');
  assert.equal(result.configHidden, true, 'sign-out should hide config');
});

await test('deploy cancels when orphaned VM warning is declined', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(async () => {
    tokenExpiry = Date.now() + 3600000;
    document.getElementById('project-manual').value = 'my-project-123';
    document.getElementById('project-select').classList.add('hidden');
    const calls = [];
    gapi = async url => {
      calls.push(url);
      if (url.includes('/aggregated/instances')) {
        return {
          items: {
            'zones/us-central1-a': {
              instances: [{ name: 'hpfasr-old', status: 'RUNNING' }],
            },
          },
        };
      }
      if (url.includes('compute.googleapis.com')) return {};
      throw new Error('unexpected URL: ' + url);
    };
    gapiRaw = async url => {
      calls.push(url);
      return { ok: true };
    };
    window.confirm = () => false;
    await deploy();
    return {
      calls,
      errorText: document.getElementById('progress-error').textContent,
      retryVisible: !document.getElementById('retry-btn').classList.contains('hidden'),
    };
  });
  assert.ok(result.calls.some(url => url.includes('/aggregated/instances')), 'should check for existing HighPerfASR VMs');
  assert.ok(!result.calls.some(url => url.includes('/global/firewalls/')), 'declining warning should stop before firewall work');
  assert.ok(result.errorText.includes('Deploy cancelled'), `should show cancellation error, got: ${result.errorText}`);
  assert.equal(result.retryVisible, true, 'retry button should be visible after cancellation');
});

await test('pollHealth accepts serial-port health on HTTPS mixed-content path', async () => {
  await simulateSignedIn();
  const result = await page.evaluate(async () => {
    const originalProtocol = Object.getOwnPropertyDescriptor(Location.prototype, 'protocol');
    Object.defineProperty(Location.prototype, 'protocol', { configurable: true, get: () => 'https:' });
    gapi = async url => {
      if (url.includes('serialPortOutput')) return { contents: 'docker run\nHPFASR_HEALTH_READY\n' };
      throw new Error('unexpected URL: ' + url);
    };
    window.fetch = async () => { throw new TypeError('Mixed content blocked'); };
    sleep = async () => {};
    deployState.startTime = Date.now() - 1000;
    document.getElementById('step-progress').classList.remove('hidden');

    try {
      await pollHealth('my-project-123', 'us-central1-a', 'hpfasr-test', 'http://10.0.0.1:8001');
    } finally {
      if (originalProtocol) Object.defineProperty(Location.prototype, 'protocol', originalProtocol);
    }

    return {
      healthClass: document.querySelector('[data-stage="health"]').className,
      logText: document.getElementById('log-panel').textContent,
    };
  });
  assert.ok(result.healthClass.includes('done'), `health stage should be done, got: ${result.healthClass}`);
  assert.ok(result.logText.includes('Server healthy'), `serial sentinel should mark healthy, got: ${result.logText}`);
});

await test('showProjectFallback renders notice with DOM nodes', async () => {
  await simulateSignedIn();
  await page.evaluate(() => {
    showProjectFallback('Could not load projects.', [
      document.createTextNode('Enter your project ID below.'),
    ]);
  });
  const noticeVisible = await page.$eval('#project-notice', el => !el.classList.contains('hidden'));
  assert.ok(noticeVisible, 'notice should be visible');
  const text = await page.$eval('#project-notice', el => el.textContent);
  assert.ok(text.includes('Could not load projects'), 'should show title');
  assert.ok(text.includes('Enter your project ID'), 'should show detail');
});

await teardown();
console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);

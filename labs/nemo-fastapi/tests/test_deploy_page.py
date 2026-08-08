import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_PAGE = REPO_ROOT / "docs" / "deploy.html"


def deploy_html() -> str:
    return DEPLOY_PAGE.read_text(encoding="utf-8")


def deploy_script() -> str:
    html = deploy_html()
    match = re.search(r"<script>(?P<script>.*)</script>", html, re.DOTALL)
    assert match, "deploy.html must contain a script block"
    return match.group("script")


def function_body(name: str) -> str:
    script = deploy_script()
    match = re.search(rf"\b(?:async\s+)?function\s+{re.escape(name)}\s*\([^)]*\)\s*\{{", script)
    assert match, f"{name}() is missing from deploy.html"

    start = match.end()
    depth = 1
    for idx in range(start, len(script)):
        if script[idx] == "{":
            depth += 1
        elif script[idx] == "}":
            depth -= 1
            if depth == 0:
                return script[start:idx]

    raise AssertionError(f"{name}() body is not balanced")


def test_deploy_button_starts_disabled_and_only_ok_status_enables_it() -> None:
    html = deploy_html()
    set_status = function_body("setProjectStatus")
    clear_status = function_body("clearProjectStatus")

    assert re.search(r'<button\b[^>]*id="deploy-btn"[^>]*\bdisabled\b', html)
    assert "deployBtn.disabled = level !== 'ok'" in set_status
    assert "document.getElementById('deploy-btn').disabled = true" in clear_status
    assert "projectValidationSeq++" in clear_status


def test_project_loading_and_selection_clear_stale_project_state() -> None:
    reset_loading = function_body("resetProjectUiForLoading")
    on_selected = function_body("onProjectSelected")
    toggle_manual = function_body("toggleManualProject")

    assert "clearProjectStatus()" in reset_loading
    assert "projectSelectionSeq++" in reset_loading
    assert "notice.replaceChildren()" in reset_loading
    assert "manual.value = ''" in reset_loading
    assert "project-manual-help').classList.add('hidden')" in reset_loading
    assert "project-manual-toggle').classList.add('hidden')" in reset_loading

    assert "sel.value === '__create__'" in on_selected
    assert "window.open('https://console.cloud.google.com/projectcreate'" in on_selected
    assert "sel.value = ''" in on_selected
    assert on_selected.count("clearProjectStatus()") >= 2

    assert "projectSelectionSeq++" in toggle_manual
    assert "clearProjectStatus()" in toggle_manual


def test_previous_project_autodetect_cannot_override_new_user_selection() -> None:
    detect_previous = function_body("detectPreviousProject")

    assert "const selectionSeq = projectSelectionSeq" in detect_previous
    assert "selectionSeq !== projectSelectionSeq" in detect_previous
    assert "sel.value" in detect_previous
    assert "sel.classList.contains('hidden')" in detect_previous
    assert "onProjectSelected(true)" in detect_previous


def test_validate_project_guards_async_results_and_billing_states() -> None:
    validate_project = function_body("validateProject")

    assert "const seq = ++projectValidationSeq" in validate_project
    assert "PROJECT_ID_RE.test(projectId)" in validate_project
    assert "setProjectStatus('error', 'Invalid project ID format" in validate_project
    assert "document.getElementById('deploy-btn').disabled = true" in validate_project
    assert validate_project.count("seq !== projectValidationSeq") >= 4

    assert "compute.googleapis.com/compute/v1/projects/${encodeURIComponent(projectId)}" in validate_project
    assert "Compute Engine API not enabled" in validate_project
    assert "setProjectStatus('warn'" in validate_project
    assert "Billing not enabled" in validate_project
    assert "billingStatus = 'unverified'" in validate_project
    assert "Billing status could not be verified" in validate_project
    assert "Project ready" in validate_project


def test_poll_health_requires_vm_sentinel_or_http_success() -> None:
    poll_health = function_body("pollHealth")

    assert "lastSerialContent.includes('HPFASR_HEALTH_READY')" in poll_health
    assert "lastSerialContent.includes('HPFASR_CONTAINER_EXITED')" in poll_health
    assert "lastSerialContent.includes('HPFASR_HEALTH_TIMEOUT')" in poll_health
    assert "if (hRes.ok)" in poll_health

    assert "const healthReady = lastSerialContent.includes('HPFASR_HEALTH_READY')" in poll_health
    assert "window.location.protocol === 'https:' && healthReady" in poll_health
    assert "lastSerialContent.includes('docker run')" in poll_health
    assert "Health check timed out after 12 minutes" in poll_health

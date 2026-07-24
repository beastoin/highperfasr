#!/usr/bin/env python3
"""Generate individual benchmark run pages for GitHub Pages."""
import json, os
from pathlib import Path

RESULTS_DIR = Path(os.path.expanduser("~/ossasr-zen/highperfasr/benchmarks/results"))
DOCS_DIR = Path(os.path.expanduser("~/ossasr-zen/highperfasr/docs/runs"))
DOCS_DIR.mkdir(parents=True, exist_ok=True)

STYLE = """\
:root{
  --ground:#080a10;--surface:#10131c;--elevated:#181c28;
  --border:#1e2436;--border-focus:#00e5a0;
  --accent:#00e5a0;--accent-dim:#00e5a033;--accent-glow:#00e5a066;
  --blue:#5a7fff;--blue-dim:#5a7fff22;
  --amber:#ffb347;--amber-dim:#ffb34722;
  --error:#ff5757;--error-dim:#ff575722;
  --text:#cdd4e0;--text2:#6b7394;--text3:#3d4560;
  --mono:'SF Mono','Cascadia Code','JetBrains Mono','Fira Code',monospace;
  --radius:6px;
}
@media(prefers-color-scheme:light){:root{
  --ground:#f4f5f8;--surface:#fff;--elevated:#eef0f4;
  --border:#d4d8e0;--border-focus:#00c98a;
  --accent:#00a87a;--accent-dim:#00a87a1a;--accent-glow:#00a87a44;
  --blue:#4466dd;--blue-dim:#4466dd15;
  --amber:#cc8800;--amber-dim:#cc880015;
  --error:#d93636;--error-dim:#d9363615;
  --text:#1a1d27;--text2:#5c6478;--text3:#8891a5;
}}
:root[data-theme="light"]{
  --ground:#f4f5f8;--surface:#fff;--elevated:#eef0f4;
  --border:#d4d8e0;--border-focus:#00c98a;
  --accent:#00a87a;--accent-dim:#00a87a1a;--accent-glow:#00a87a44;
  --blue:#4466dd;--blue-dim:#4466dd15;
  --amber:#cc8800;--amber-dim:#cc880015;
  --error:#d93636;--error-dim:#d9363615;
  --text:#1a1d27;--text2:#5c6478;--text3:#8891a5;
}
:root[data-theme="dark"]{
  --ground:#080a10;--surface:#10131c;--elevated:#181c28;
  --border:#1e2436;--border-focus:#00e5a0;
  --accent:#00e5a0;--accent-dim:#00e5a033;--accent-glow:#00e5a066;
  --blue:#5a7fff;--blue-dim:#5a7fff22;
  --amber:#ffb347;--amber-dim:#ffb34722;
  --error:#ff5757;--error-dim:#ff575722;
  --text:#cdd4e0;--text2:#6b7394;--text3:#3d4560;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--ground);color:var(--text);font-size:13px;line-height:1.5;min-height:100vh}
.topbar{display:flex;align-items:center;gap:16px;padding:12px 24px;border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--ground);z-index:10}
.logo{font-family:var(--mono);font-size:16px;font-weight:700;color:var(--text)}.logo em{color:var(--accent);font-style:normal}
.topbar a{color:var(--accent);font-family:var(--mono);font-size:12px;text-decoration:none}
.topbar a:hover{text-decoration:underline}
.container{max-width:1200px;margin:0 auto;padding:24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:16px}
.card-title{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--text2);margin-bottom:12px}
.metrics{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px}
.metric{background:var(--elevated);border:1px solid var(--border);border-radius:var(--radius);padding:14px 18px;min-width:120px;text-align:center;flex:1}
.metric-val{font-family:var(--mono);font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}
.metric-label{font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--text2);margin-top:4px}
.metric-note{font-size:10px;color:var(--text3);margin-top:2px;font-family:var(--mono)}
.c-accent{color:var(--accent)}.c-blue{color:var(--blue)}.c-amber{color:var(--amber)}.c-error{color:var(--error)}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums}
th{text-align:left;padding:8px 10px;border-bottom:2px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--text2)}
td{padding:6px 10px;border-bottom:1px solid var(--border)}
.text-r{text-align:right}.text-c{text-align:center}
.table-wrap{overflow-x:auto;margin-top:8px}
.info-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px;font-family:var(--mono);font-size:12px}
.info-item{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)}
.info-label{color:var(--text2)}
.chart-wrap{position:relative;height:240px;margin:16px 0}
.chart-wrap canvas{width:100%;height:100%}
.chart-tooltip{position:absolute;background:var(--elevated);border:1px solid var(--border);border-radius:4px;padding:6px 10px;font-family:var(--mono);font-size:11px;pointer-events:none;opacity:0;transition:opacity .15s;white-space:nowrap;z-index:5}
.footer{border-top:1px solid var(--border);padding:20px 24px;text-align:center;color:var(--text3);font-size:12px;margin-top:40px}
.footer a{color:var(--text2);text-decoration:none;font-family:var(--mono)}
.footer a:hover{color:var(--accent)}
.nav-links{display:flex;flex-wrap:wrap;gap:8px;font-family:var(--mono);font-size:11px;margin-bottom:16px}
.nav-links a{color:var(--accent);text-decoration:none;padding:2px 6px;border:1px solid var(--border);border-radius:4px}
.nav-links a:hover{border-color:var(--accent)}
.nav-links .current{color:var(--text);font-weight:700;padding:2px 6px;border:1px solid var(--accent);border-radius:4px;background:var(--accent-dim)}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-family:var(--mono);font-size:10px;font-weight:700}
.badge-batch{background:var(--accent-dim);color:var(--accent)}
.badge-stream{background:var(--blue-dim);color:var(--blue)}
.repro{background:var(--elevated);border-radius:var(--radius);padding:12px 16px;font-family:var(--mono);font-size:11px;overflow-x:auto;white-space:pre-wrap;word-break:break-all;color:var(--text2);margin-top:8px;border:1px solid var(--border)}
.legend{display:flex;gap:16px;font-family:var(--mono);font-size:11px;margin-top:8px;justify-content:center}
.legend-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:middle}
"""

CHART_JS = """\
function drawBarLine(id, data, opts) {
  const wrap = document.getElementById(id);
  if (!wrap) return;
  const canvas = wrap.querySelector('canvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {t:20, r:60, b:36, l:56};
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const cs = getComputedStyle(document.documentElement);
  const accent = cs.getPropertyValue('--accent').trim();
  const blue = cs.getPropertyValue('--blue').trim();
  const gridC = cs.getPropertyValue('--border').trim();
  const textC = cs.getPropertyValue('--text2').trim();
  ctx.clearRect(0, 0, W, H);

  const n = data.x.length;
  const barColor = opts.barColor === 'blue' ? blue : accent;
  const lineColor = opts.lineColor === 'blue' ? blue : accent;
  const maxBar = Math.max(...data.bars) * 1.15 || 1;
  const maxLine = Math.max(...data.line) * 1.15 || 1;
  const gap = plotW / n;
  const barW = Math.min(gap * 0.45, 36);

  for (let i = 0; i <= 4; i++) {
    const y = pad.t + plotH * (1 - i/4);
    ctx.strokeStyle = gridC; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
    ctx.fillStyle = textC; ctx.font = '10px system-ui';
    ctx.textAlign = 'right'; ctx.fillText((maxBar*i/4).toFixed(opts.barDecimals||1), pad.l-6, y+3);
    ctx.textAlign = 'left';
    const rv = maxLine*i/4;
    ctx.fillText(opts.lineSuffix ? rv.toFixed(0)+opts.lineSuffix : rv.toFixed(1), W-pad.r+6, y+3);
  }

  data.x.forEach((lbl, i) => {
    const bx = pad.l + gap*i + (gap-barW)/2;
    const bh = (data.bars[i]/maxBar)*plotH;
    ctx.fillStyle = barColor+'44'; ctx.fillRect(bx, pad.t+plotH-bh, barW, bh);
    ctx.fillStyle = barColor; ctx.fillRect(bx, pad.t+plotH-bh, barW, 2);
    ctx.fillStyle = textC; ctx.font='10px system-ui'; ctx.textAlign='center';
    ctx.fillText(lbl, pad.l+gap*i+gap/2, H-pad.b+14);
  });

  ctx.beginPath(); ctx.strokeStyle = lineColor; ctx.lineWidth = 2;
  data.x.forEach((_, i) => {
    const x = pad.l+gap*i+gap/2, y = pad.t+plotH*(1-data.line[i]/maxLine);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  data.x.forEach((_, i) => {
    const x = pad.l+gap*i+gap/2, y = pad.t+plotH*(1-data.line[i]/maxLine);
    ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fillStyle=lineColor; ctx.fill();
  });

  ctx.fillStyle=textC; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText(opts.xLabel||'Concurrency', pad.l+plotW/2, H-2);
  ctx.save(); ctx.translate(12,pad.t+plotH/2); ctx.rotate(-Math.PI/2); ctx.fillText(opts.barLabel||'',0,0); ctx.restore();
  ctx.save(); ctx.translate(W-8,pad.t+plotH/2); ctx.rotate(-Math.PI/2); ctx.fillText(opts.lineLabel||'',0,0); ctx.restore();

  const tooltip = wrap.querySelector('.chart-tooltip');
  if (tooltip) {
    canvas.onmousemove = e => {
      const idx = Math.floor((e.offsetX-pad.l)/gap);
      if (idx>=0 && idx<n && e.offsetX>=pad.l) {
        tooltip.style.opacity='1';
        let html = (opts.xLabel||'c') + '=' + data.x[idx];
        html += '<br>'+(opts.barLabel||'bar')+': '+data.bars[idx];
        html += '<br>'+(opts.lineLabel||'line')+': '+data.line[idx];
        if (data.extra) data.extra.forEach(e2 => { if(e2.vals[idx]!=null) html+='<br>'+e2.label+': '+e2.vals[idx]; });
        tooltip.innerHTML = html;
        tooltip.style.left = Math.min(pad.l+gap*idx+gap/2+10, W-150) + 'px';
        tooltip.style.top = '10px';
      } else tooltip.style.opacity='0';
    };
    canvas.onmouseleave = () => tooltip.style.opacity='0';
  }
}

function drawDualLine(id, data, opts) {
  const wrap = document.getElementById(id);
  if (!wrap) return;
  const canvas = wrap.querySelector('canvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = {t:20, r:16, b:36, l:56};
  const plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;
  const cs = getComputedStyle(document.documentElement);
  const accent = cs.getPropertyValue('--accent').trim();
  const blue = cs.getPropertyValue('--blue').trim();
  const gridC = cs.getPropertyValue('--border').trim();
  const textC = cs.getPropertyValue('--text2').trim();
  ctx.clearRect(0, 0, W, H);

  const n = data.x.length;
  const maxY = Math.max(...data.line1, ...data.line2) * 1.15 || 1;

  for (let i = 0; i <= 4; i++) {
    const y = pad.t + plotH * (1 - i/4);
    ctx.strokeStyle = gridC; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
    ctx.fillStyle = textC; ctx.font = '10px system-ui';
    ctx.textAlign = 'right'; ctx.fillText((maxY*i/4).toFixed(0) + (opts.ySuffix||''), pad.l-6, y+3);
  }

  const spacing = n > 1 ? plotW/(n-1) : plotW;
  data.x.forEach((lbl, i) => {
    ctx.fillStyle = textC; ctx.font='10px system-ui'; ctx.textAlign='center';
    ctx.fillText(lbl, pad.l+spacing*i, H-pad.b+14);
  });

  [
    {vals: data.line1, color: accent},
    {vals: data.line2, color: blue}
  ].forEach(series => {
    ctx.beginPath(); ctx.strokeStyle = series.color; ctx.lineWidth = 2;
    series.vals.forEach((v, i) => {
      const x = pad.l+spacing*i, y = pad.t+plotH*(1-v/maxY);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
    series.vals.forEach((v, i) => {
      const x = pad.l+spacing*i, y = pad.t+plotH*(1-v/maxY);
      ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fillStyle=series.color; ctx.fill();
    });
  });

  ctx.fillStyle=textC; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText(opts.xLabel||'Concurrency', pad.l+plotW/2, H-2);

  const tooltip = wrap.querySelector('.chart-tooltip');
  if (tooltip) {
    canvas.onmousemove = e => {
      const idx = Math.round((e.offsetX-pad.l)/spacing);
      if (idx>=0 && idx<n && e.offsetX>=pad.l-10) {
        tooltip.style.opacity='1';
        tooltip.innerHTML = 'c='+data.x[idx]+'<br>'+(opts.label1||'A')+': '+data.line1[idx]+(opts.ySuffix||'')+'<br>'+(opts.label2||'B')+': '+data.line2[idx]+(opts.ySuffix||'');
        tooltip.style.left = Math.min(pad.l+spacing*idx+10, W-150)+'px';
        tooltip.style.top = '10px';
      } else tooltip.style.opacity='0';
    };
    canvas.onmouseleave = () => tooltip.style.opacity='0';
  }
}

const _ro = new ResizeObserver(() => {
  document.querySelectorAll('.chart-wrap').forEach(w => { if (w._drawFn) w._drawFn(); });
});
document.querySelectorAll('.chart-wrap').forEach(w => _ro.observe(w));
"""

def esc(s):
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def load_json(p):
    return json.loads(p.read_text()) if p.exists() else None

def fmt_num(v, suffix=""):
    if v is None: return "—"
    return f"{v}{suffix}"

ALL_RUNS = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])

def nav_html(current):
    parts = []
    for r in ALL_RUNS:
        if r == current:
            parts.append(f'<span class="current">{r}</span>')
        else:
            parts.append(f'<a href="{r}.html">{r}</a>')
    return '<div class="nav-links">' + ''.join(parts) + '</div>'

def info_card(result):
    if not result: return ""
    sut = result.get("sut", {})
    hw = result.get("hardware", {})
    sw = result.get("software", {})
    env = result.get("environment", {})
    ds = result.get("dataset", {})
    items = [
        ("Model", sut.get("model_name")),
        ("Params", sut.get("model_params")),
        ("GPU", hw.get("gpu")),
        ("GPU Memory", f'{hw.get("gpu_memory_gb")} GB' if hw.get("gpu_memory_gb") else None),
        ("CPU", hw.get("cpu")),
        ("RAM", f'{hw.get("ram_gb")} GB' if hw.get("ram_gb") else None),
        ("CUDA", sw.get("cuda_version")),
        ("PyTorch", sw.get("pytorch_version")),
        ("Framework", sw.get("framework_version")),
        ("Container", sw.get("container_image")),
        ("Dataset", ds.get("name")),
        ("Split", ds.get("split")),
        ("Files", ds.get("num_files")),
        ("Audio Hours", ds.get("total_audio_hours")),
        ("Provider", env.get("provider")),
        ("Region", env.get("region")),
        ("Instance", env.get("instance_type")),
    ]
    items = [(l,v) for l,v in items if v]
    if not items: return ""
    html = '<div class="card"><div class="card-title">System Under Test</div><div class="info-grid">'
    for label, val in items:
        html += f'<div class="info-item"><span class="info-label">{label}</span><span>{esc(str(val))}</span></div>'
    html += '</div></div>'
    return html

def repro_card(result):
    if not result: return ""
    r = result.get("reproduction", {})
    cmd = r.get("command", "")
    notes = r.get("notes", "")
    if not cmd and not notes: return ""
    html = '<div class="card"><div class="card-title">Reproduction</div>'
    if cmd: html += f'<div class="repro">{esc(cmd)}</div>'
    if notes: html += f'<div style="margin-top:8px;font-size:12px;color:var(--text2)">{esc(notes)}</div>'
    html += '</div>'
    return html

def cost_card(result):
    if not result: return ""
    c = result.get("cost", {})
    if not c: return ""
    items = [
        ("Pricing", c.get("pricing_model")),
        ("Instance $/hr", f'${c.get("node_usd_per_hour")}' if c.get("node_usd_per_hour") else None),
        ("$/audio-hour", f'${c.get("usd_per_audio_hour")}' if c.get("usd_per_audio_hour") else None),
        ("Wall Clock", f'{c.get("wall_clock_seconds")}s' if c.get("wall_clock_seconds") else None),
    ]
    items = [(l,v) for l,v in items if v]
    if not items: return ""
    html = '<div class="card"><div class="card-title">Cost</div><div class="info-grid">'
    for l,v in items:
        html += f'<div class="info-item"><span class="info-label">{l}</span><span>{esc(str(v))}</span></div>'
    html += '</div></div>'
    return html


def build_standard_page(run_id, result_dir):
    result = load_json(result_dir / "result.json")
    sweep = load_json(result_dir / "concurrency-sweep.json")
    if not result and not sweep: return None

    sut = (result or {}).get("sut", {})
    hw = (result or {}).get("hardware", {})
    scenario = (result or {}).get("scenario", {})
    quality = (result or {}).get("quality", {})
    perf = (result or {}).get("performance", {})
    rel = (result or {}).get("reliability", {})
    ds = (result or {}).get("dataset", {})

    mode_raw = scenario.get("mode", "offline")
    is_stream = "stream" in run_id or "stream" in mode_raw
    badge = ("Streaming", "badge-stream") if is_stream else ("Batch", "badge-batch")
    gpu = hw.get("gpu", "GPU")
    model = sut.get("model_name", "")
    dataset = f'{ds.get("name","")} {ds.get("split","")}'.strip()
    timestamp = (result or {}).get("generated_at", "")
    date = timestamp[:10]
    git_sha = sut.get("git_sha", "")
    wer = quality.get("wer")
    rtfx = perf.get("rtfx")
    rps = perf.get("rps") or perf.get("sessions_per_min")
    max_c = perf.get("max_concurrent_streams")
    failures = rel.get("failed", 0)
    total = rel.get("total_requests", 0)
    rps_label = "Sess/min" if is_stream else "Peak RPS"

    levels = (sweep or {}).get("levels", [])
    chart_init = ""
    chart_html = ""
    sweep_table = ""

    if levels:
        cid = f"chart-{run_id}"
        chart_html = f'<div class="chart-wrap" id="{cid}"><canvas></canvas><div class="chart-tooltip"></div></div>'

        if is_stream:
            cd = json.dumps({
                "x": [str(l["concurrency"]) for l in levels],
                "bars": [l.get("rtfx", 0) for l in levels],
                "line": [l.get("sessions_per_min", 0) for l in levels],
            })
            chart_init = f"var d1={cd};var w1=document.getElementById('{cid}');w1._drawFn=function(){{drawBarLine('{cid}',d1,{{barColor:'blue',barLabel:'RTFx',lineLabel:'Sess/min',barDecimals:0}})}};requestAnimationFrame(w1._drawFn);_ro.observe(w1);"
            sweep_table = '<div class="table-wrap"><table><thead><tr><th class="text-c">C</th><th class="text-r">RTFx</th><th class="text-r">Sess/min</th><th class="text-c">Failures</th></tr></thead><tbody>'
            for l in levels:
                f = l.get("failures", 0)
                fc = "c-error" if f else "c-accent"
                sweep_table += f'<tr><td class="text-c">{l["concurrency"]}</td><td class="text-r">{l.get("rtfx","—")}x</td><td class="text-r">{l.get("sessions_per_min","—")}</td><td class="text-c {fc}">{f}</td></tr>'
            sweep_table += '</tbody></table></div>'
        else:
            cd = json.dumps({
                "x": [str(l["concurrency"]) for l in levels],
                "bars": [l.get("rps", 0) for l in levels],
                "line": [l.get("rtfx", 0) for l in levels],
                "extra": [
                    {"label": "p50", "vals": [f'{l.get("p50_ms","")}ms' for l in levels]},
                    {"label": "p99", "vals": [f'{l.get("p99_ms","")}ms' for l in levels]},
                ]
            })
            chart_init = f"var d1={cd};var w1=document.getElementById('{cid}');w1._drawFn=function(){{drawBarLine('{cid}',d1,{{barLabel:'RPS',lineLabel:'RTFx',lineSuffix:'x',barDecimals:1}})}};requestAnimationFrame(w1._drawFn);_ro.observe(w1);"
            sweep_table = '<div class="table-wrap"><table><thead><tr><th class="text-c">C</th><th class="text-r">RPS</th><th class="text-r">RTFx</th><th class="text-r">p50</th><th class="text-r">p99</th><th class="text-c">Failures</th></tr></thead><tbody>'
            for l in levels:
                f = l.get("failures", 0)
                fc = "c-error" if f else "c-accent"
                sweep_table += f'<tr><td class="text-c">{l["concurrency"]}</td><td class="text-r">{l.get("rps","—")}</td><td class="text-r">{l.get("rtfx","—")}x</td><td class="text-r">{l.get("p50_ms","—")}ms</td><td class="text-r">{l.get("p99_ms","—")}ms</td><td class="text-c {fc}">{f}</td></tr>'
            sweep_table += '</tbody></table></div>'

    return assemble_page(run_id, badge, gpu, model, dataset, date, scenario.get("description",""),
        f"""<div class="metrics">
          <div class="metric"><div class="metric-val {"c-accent" if wer is not None and wer<4 else "c-blue"}">{fmt_num(wer,'%')}</div><div class="metric-label">WER</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(rps)}</div><div class="metric-label">{rps_label}</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(rtfx,'x')}</div><div class="metric-label">RTFx</div></div>
          <div class="metric"><div class="metric-val {"c-accent" if failures==0 else "c-error"}">{failures}</div><div class="metric-label">Failures</div><div class="metric-note">of {total}</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(max_c)}</div><div class="metric-label">Max Concurrency</div></div>
        </div>""",
        f'<div class="card"><div class="card-title">Concurrency Sweep</div>{chart_html}{sweep_table}</div>',
        info_card(result) + repro_card(result) + cost_card(result),
        chart_init, git_sha=git_sha, timestamp=timestamp)


def build_duration_page(run_id, result_dir):
    result = load_json(result_dir / "result.json")
    raw = load_json(result_dir / "raw.json")
    if not result and not raw: return None

    sut = (result or {}).get("sut", {})
    hw = (result or {}).get("hardware", {})
    scenario = (result or {}).get("scenario", {})
    perf = (result or {}).get("performance", {})
    rel = (result or {}).get("reliability", {})
    ds = (result or {}).get("dataset", {})

    gpu = hw.get("gpu", "GPU")
    model = sut.get("model_name", "")
    dataset = f'{ds.get("name","")} {ds.get("split","")}'.strip()
    timestamp = (result or {}).get("generated_at", "")
    date = timestamp[:10]
    git_sha = sut.get("git_sha", "")
    rtfx = perf.get("rtfx")
    rps = perf.get("rps")
    failures = rel.get("failed", 0)
    total = rel.get("total_requests", 0)

    dur_table = ""
    dur_chart_init = ""
    dur_chart_html = ""
    if raw and "durations" in raw:
        durs = raw["durations"]
        dur_table = '<div class="table-wrap"><table><thead><tr><th>Duration</th><th class="text-r">Max Safe C</th><th class="text-r">Peak RPS</th><th class="text-r">Peak RTFx</th><th class="text-r">Peak C</th></tr></thead><tbody>'
        xs, bars, line = [], [], []
        for d in durs:
            dur_s = d["duration_sec"]
            dur_table += f'<tr><td>{dur_s}s</td><td class="text-r">{d.get("max_safe_concurrency","—")}</td><td class="text-r">{d.get("peak_rps","—")}</td><td class="text-r">{d.get("peak_rtfx","—")}x</td><td class="text-r">{d.get("peak_concurrency","—")}</td></tr>'
            xs.append(f'{dur_s}s')
            bars.append(d.get("peak_rps", 0))
            line.append(d.get("peak_rtfx", 0))
        dur_table += '</tbody></table></div>'

        cid = f"chart-{run_id}"
        dur_chart_html = f'<div class="chart-wrap" id="{cid}"><canvas></canvas><div class="chart-tooltip"></div></div>'
        cd = json.dumps({"x": xs, "bars": bars, "line": line})
        dur_chart_init = f"var dd={cd};var wd=document.getElementById('{cid}');wd._drawFn=function(){{drawBarLine('{cid}',dd,{{xLabel:'Audio Duration',barLabel:'Peak RPS',lineLabel:'Peak RTFx',lineSuffix:'x',barDecimals:1}})}};requestAnimationFrame(wd._drawFn);_ro.observe(wd);"

    return assemble_page(run_id, ("Batch", "badge-batch"), gpu, model, dataset, date, scenario.get("description",""),
        f"""<div class="metrics">
          <div class="metric"><div class="metric-val">{fmt_num(rps)}</div><div class="metric-label">Peak RPS</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(rtfx,'x')}</div><div class="metric-label">RTFx</div></div>
          <div class="metric"><div class="metric-val {"c-accent" if failures==0 else "c-error"}">{failures}</div><div class="metric-label">Failures</div><div class="metric-note">of {total}</div></div>
        </div>""",
        f'<div class="card"><div class="card-title">Duration Scaling</div>{dur_chart_html}{dur_table}</div>',
        info_card(result) + repro_card(result) + cost_card(result),
        dur_chart_init, git_sha=git_sha, timestamp=timestamp)


def build_dockerfile_stream_page(run_id, result_dir):
    sweep = load_json(result_dir / "sweep.json")
    soak = load_json(result_dir / "soak.json")
    if not sweep and not soak: return None

    git_sha = (sweep or {}).get("system", {}).get("git_sha", "")
    timestamp = (sweep or {}).get("timestamp", "")
    wer = sweep.get("wer", {}).get("corpus_wer_pct") if sweep else None
    cs_data = sweep.get("concurrency_sweep", []) if sweep else []
    sustained = sweep.get("sustained_load", {}) if sweep else {}

    soak_dur = (soak.get("durations", [{}])[0]) if soak else {}
    soak_streams = soak_dur.get("total_streams", 0)
    soak_failures = soak_dur.get("failures", 0)
    soak_vram_growth = soak_dur.get("vram_growth_mb", 0)
    soak_duration_s = soak_dur.get("actual_duration_s", 0)
    soak_wer = soak_dur.get("wer_pct")

    chart_init = ""
    chart_html = ""
    if cs_data:
        cid = f"chart-{run_id}"
        chart_html = f'<div class="chart-wrap" id="{cid}"><canvas></canvas><div class="chart-tooltip"></div></div>'
        cd = json.dumps({
            "x": [str(l["concurrency"]) for l in cs_data],
            "bars": [l.get("rtfx", 0) for l in cs_data],
            "line": [l.get("sess_per_min", 0) for l in cs_data],
        })
        chart_init = f"var ds={cd};var ws=document.getElementById('{cid}');ws._drawFn=function(){{drawBarLine('{cid}',ds,{{barColor:'blue',barLabel:'RTFx',lineLabel:'Sess/min',barDecimals:0}})}};requestAnimationFrame(ws._drawFn);_ro.observe(ws);"

    sweep_table = ""
    if cs_data:
        sweep_table = '<div class="table-wrap"><table><thead><tr><th class="text-c">C</th><th class="text-r">RTFx</th><th class="text-r">Sess/min</th><th class="text-r">p50</th><th class="text-r">p99</th><th class="text-c">Fail</th></tr></thead><tbody>'
        for l in cs_data:
            f = l.get("failures", 0)
            fc = "c-error" if f else "c-accent"
            sweep_table += f'<tr><td class="text-c">{l["concurrency"]}</td><td class="text-r">{l.get("rtfx","—")}x</td><td class="text-r">{l.get("sess_per_min","—")}</td><td class="text-r">{l.get("p50_s","—")}s</td><td class="text-r">{l.get("p99_s","—")}s</td><td class="text-c {fc}">{f}</td></tr>'
        sweep_table += '</tbody></table></div>'

    sustained_html = ""
    if sustained:
        sf = sustained.get("failures", 0)
        sfc = "c-error" if sf else "c-accent"
        sustained_html = f"""<div class="card"><div class="card-title">Sustained Load</div>
        <div class="metrics">
          <div class="metric"><div class="metric-val c-blue">{sustained.get("concurrency","—")}</div><div class="metric-label">Concurrency</div></div>
          <div class="metric"><div class="metric-val">{sustained.get("sess_per_min","—")}</div><div class="metric-label">Sess/min</div></div>
          <div class="metric"><div class="metric-val">{sustained.get("rtfx","—")}x</div><div class="metric-label">RTFx</div></div>
          <div class="metric"><div class="metric-val {sfc}">{sf}</div><div class="metric-label">Failures</div><div class="metric-note">{sustained.get("rounds","?")} rounds, {sustained.get("total",0)} total</div></div>
        </div></div>"""

    soak_html = ""
    if soak_dur:
        sfc2 = "c-error" if soak_failures else "c-accent"
        vc = "c-accent" if soak_vram_growth < 100 else "c-error"
        soak_html = f"""<div class="card"><div class="card-title">Soak Test ({int(soak_duration_s)}s)</div>
        <div class="metrics">
          <div class="metric"><div class="metric-val">{soak_streams}</div><div class="metric-label">Streams</div></div>
          <div class="metric"><div class="metric-val {sfc2}">{soak_failures}</div><div class="metric-label">Failures</div></div>
          <div class="metric"><div class="metric-val {vc}">{soak_vram_growth} MB</div><div class="metric-label">VRAM Growth</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(soak_wer,'%')}</div><div class="metric-label">WER</div></div>
        </div></div>"""

    peak_rtfx = max([l.get("rtfx",0) for l in cs_data]) if cs_data else None
    peak_sess = max([l.get("sess_per_min",0) for l in cs_data]) if cs_data else None

    return assemble_page(run_id, ("Streaming", "badge-stream"), "NVIDIA L4", "", "LibriSpeech test-clean",
        timestamp[:10] if timestamp else "",
        "Dockerfile-built streaming server: concurrency sweep + soak test",
        f"""<div class="metrics">
          <div class="metric"><div class="metric-val c-accent">{fmt_num(wer,'%')}</div><div class="metric-label">WER</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(peak_sess)}</div><div class="metric-label">Peak Sess/min</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(peak_rtfx,'x')}</div><div class="metric-label">Peak RTFx</div></div>
          <div class="metric"><div class="metric-val {"c-accent" if soak_vram_growth<100 else "c-error"}">{soak_vram_growth} MB</div><div class="metric-label">VRAM Growth</div></div>
        </div>""",
        f'<div class="card"><div class="card-title">Concurrency Sweep</div>{chart_html}{sweep_table}</div>' + sustained_html + soak_html,
        "", chart_init, git_sha=git_sha, timestamp=timestamp)


def build_comparison_page(run_id, result_dir):
    hpfasr = load_json(result_dir / "hpfasr-sweep.json")
    nemo = load_json(result_dir / "nemo-fork-sweep.json")
    if not hpfasr and not nemo: return None

    git_sha = (hpfasr or {}).get("system", {}).get("git_sha", "")
    timestamp = (hpfasr or {}).get("timestamp", "")
    hcs = hpfasr.get("concurrency_sweep", []) if hpfasr else []
    ncs = nemo.get("concurrency_sweep", []) if nemo else []

    all_c = sorted(set([l["concurrency"] for l in hcs] + [l["concurrency"] for l in ncs]))
    h_map = {l["concurrency"]: l for l in hcs}
    n_map = {l["concurrency"]: l for l in ncs}

    comp_table = '<div class="table-wrap"><table><thead><tr><th class="text-c">C</th><th class="text-r">HPFASR RTFx</th><th class="text-r">NeMo RTFx</th><th class="text-r">HPFASR Sess/min</th><th class="text-r">NeMo Sess/min</th><th class="text-r">Delta</th></tr></thead><tbody>'
    for c in all_c:
        h = h_map.get(c, {})
        n = n_map.get(c, {})
        hr = h.get("rtfx", 0)
        nr = n.get("rtfx", 0)
        delta = f"+{((hr/nr-1)*100):.0f}%" if nr > 0 and hr > 0 else "—"
        dc = "c-accent" if hr >= nr else "c-error"
        comp_table += f'<tr><td class="text-c">{c}</td><td class="text-r">{h.get("rtfx","—")}x</td><td class="text-r">{n.get("rtfx","—")}x</td><td class="text-r">{h.get("sess_per_min","—")}</td><td class="text-r">{n.get("sess_per_min","—")}</td><td class="text-r {dc}">{delta}</td></tr>'
    comp_table += '</tbody></table></div>'

    cid = f"chart-{run_id}"
    chart_html = f"""<div class="chart-wrap" id="{cid}"><canvas></canvas><div class="chart-tooltip"></div></div>
    <div class="legend"><span><span class="legend-dot" style="background:var(--accent)"></span>HighPerfASR</span><span><span class="legend-dot" style="background:var(--blue)"></span>NeMo Fork</span></div>"""
    cd = json.dumps({
        "x": [str(c) for c in all_c],
        "line1": [h_map.get(c, {}).get("rtfx", 0) for c in all_c],
        "line2": [n_map.get(c, {}).get("rtfx", 0) for c in all_c],
    })
    chart_init = f"var dc={cd};var wc=document.getElementById('{cid}');wc._drawFn=function(){{drawDualLine('{cid}',dc,{{xLabel:'Concurrency',ySuffix:'x',label1:'HPFASR',label2:'NeMo Fork'}})}};requestAnimationFrame(wc._drawFn);_ro.observe(wc);"

    h_wer = hpfasr.get("wer",{}).get("corpus_wer_pct") if hpfasr else None
    h_peak = max([l.get("rtfx",0) for l in hcs]) if hcs else 0
    n_peak = max([l.get("rtfx",0) for l in ncs]) if ncs else 0
    h_peak_sess = max([l.get("sess_per_min",0) for l in hcs]) if hcs else 0

    return assemble_page(run_id, ("Streaming", "badge-stream"), "NVIDIA L4", "", "LibriSpeech test-clean",
        timestamp[:10] if timestamp else "",
        "Head-to-head: HighPerfASR vs NeMo Fork streaming throughput",
        f"""<div class="metrics">
          <div class="metric"><div class="metric-val c-accent">{h_peak}x</div><div class="metric-label">HPFASR Peak RTFx</div></div>
          <div class="metric"><div class="metric-val c-blue">{n_peak}x</div><div class="metric-label">NeMo Peak RTFx</div></div>
          <div class="metric"><div class="metric-val">{h_peak_sess}</div><div class="metric-label">HPFASR Peak Sess/min</div></div>
          <div class="metric"><div class="metric-val c-accent">{fmt_num(h_wer,'%')}</div><div class="metric-label">WER (both)</div></div>
        </div>""",
        f'<div class="card"><div class="card-title">RTFx Comparison</div>{chart_html}</div><div class="card"><div class="card-title">Detailed Comparison</div>{comp_table}</div>',
        "", chart_init, git_sha=git_sha, timestamp=timestamp)


def assemble_page(run_id, badge, gpu, model, dataset, date, desc, metrics_html, body_html, extra_html, chart_init, git_sha="", timestamp=""):
    commit_html = ""
    if git_sha:
        commit_url = f"https://github.com/beastoin/highperfasr/commit/{git_sha}"
        commit_html = f' &middot; <a href="{commit_url}" style="color:var(--accent);font-family:var(--mono);font-size:12px;text-decoration:none" title="Code commit that produced this benchmark"><code>{esc(git_sha)}</code></a>'
    ts_display = timestamp or date
    return f"""<title>{run_id} — HighPerfASR Benchmark</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{STYLE}</style>
<div class="topbar">
  <div class="logo">High<em>Perf</em>ASR</div>
  <span style="color:var(--text2);font-family:var(--mono);font-size:12px">{esc(run_id)}</span>
  <div style="flex:1"></div>
  <a href="../index.html">&larr; All Results</a>
  <a href="../dashboard.html">Dashboard</a>
  <a href="https://github.com/beastoin/highperfasr">GitHub</a>
</div>
<div class="container">
  <div style="margin-bottom:20px">
    <h1 style="font-family:var(--mono);font-size:18px;font-weight:700;margin-bottom:4px">{esc(run_id)} <span class="badge {badge[1]}">{badge[0]}</span></h1>
    <div style="font-family:var(--mono);font-size:12px;color:var(--text2)">{esc(model)}{' &middot; ' if model else ''}{esc(dataset)} &middot; {esc(gpu)} &middot; {esc(ts_display)}{commit_html}</div>
    {'<div style="font-size:12px;color:var(--text3);margin-top:4px">'+esc(desc)+'</div>' if desc else ''}
  </div>
  {metrics_html}
  {body_html}
  {extra_html}
  <div style="margin-top:24px">{nav_html(run_id)}</div>
</div>
<div class="footer">
  <a href="../index.html">All Results</a> &middot;
  <a href="../dashboard.html">Interactive Dashboard</a> &middot;
  <a href="https://github.com/beastoin/highperfasr">Repository</a>
</div>
<script>
{CHART_JS}
{chart_init}
</script>"""


def build_live_page(run_id, result_dir):
    """Builder for v1alpha2-live schema reports (auto-published by --publish flag)."""
    report = load_json(result_dir / "report.json")
    sweep = load_json(result_dir / "concurrency-sweep.json")
    if not report: return None

    sys_info = report.get("system", {})
    is_stream = "stream" in run_id or "stream" in report.get("benchmark", "").lower()
    badge = ("Streaming", "badge-stream") if is_stream else ("Batch", "badge-batch")
    gpu = sys_info.get("gpu", "GPU")
    git_sha = sys_info.get("git_sha", "")
    timestamp = report.get("timestamp", "")
    dataset = report.get("dataset", "")
    wer_info = report.get("wer", {})
    wer = wer_info.get("corpus_wer_pct")
    norm = wer_info.get("normalization", "")
    sustained = report.get("sustained_load", {})
    cs = report.get("concurrency_sweep", [])
    levels = (sweep or {}).get("levels", []) if sweep else []
    if not levels and cs:
        levels = cs

    peak_rps = max([l.get("rps", 0) for l in levels], default=0)
    peak_rtfx = max([l.get("rtfx", 0) for l in levels], default=0)
    peak_sess = max([l.get("sessions_per_min", l.get("sess_per_min", 0)) for l in levels], default=0)
    total_failures = sum(l.get("failures", 0) for l in levels)

    chart_init = ""
    chart_html = ""
    sweep_table = ""
    if levels:
        cid = f"chart-{run_id}"
        chart_html = f'<div class="chart-wrap" id="{cid}"><canvas></canvas><div class="chart-tooltip"></div></div>'
        if is_stream:
            cd = json.dumps({
                "x": [str(l["concurrency"]) for l in levels],
                "bars": [l.get("rtfx", 0) for l in levels],
                "line": [l.get("sessions_per_min", l.get("sess_per_min", 0)) for l in levels],
            })
            chart_init = f"var d1={cd};var w1=document.getElementById('{cid}');w1._drawFn=function(){{drawBarLine('{cid}',d1,{{barColor:'blue',barLabel:'RTFx',lineLabel:'Sess/min',barDecimals:0}})}};requestAnimationFrame(w1._drawFn);_ro.observe(w1);"
            sweep_table = '<div class="table-wrap"><table><thead><tr><th class="text-c">C</th><th class="text-r">RTFx</th><th class="text-r">Sess/min</th><th class="text-r">p50</th><th class="text-r">p99</th><th class="text-c">Failures</th></tr></thead><tbody>'
            for l in levels:
                f = l.get("failures", 0)
                fc = "c-error" if f else "c-accent"
                p50 = l.get("p50_ms", l.get("p50_s", "—"))
                p99 = l.get("p99_ms", l.get("p99_s", "—"))
                p50_s = f'{p50}ms' if isinstance(p50, (int, float)) else str(p50)
                p99_s = f'{p99}ms' if isinstance(p99, (int, float)) else str(p99)
                sweep_table += f'<tr><td class="text-c">{l["concurrency"]}</td><td class="text-r">{l.get("rtfx","—")}x</td><td class="text-r">{l.get("sessions_per_min", l.get("sess_per_min","—"))}</td><td class="text-r">{p50_s}</td><td class="text-r">{p99_s}</td><td class="text-c {fc}">{f}</td></tr>'
            sweep_table += '</tbody></table></div>'
        else:
            cd = json.dumps({
                "x": [str(l["concurrency"]) for l in levels],
                "bars": [l.get("rps", 0) for l in levels],
                "line": [l.get("rtfx", 0) for l in levels],
                "extra": [
                    {"label": "p50", "vals": [f'{l.get("p50_ms","")}ms' for l in levels]},
                    {"label": "p99", "vals": [f'{l.get("p99_ms","")}ms' for l in levels]},
                ]
            })
            chart_init = f"var d1={cd};var w1=document.getElementById('{cid}');w1._drawFn=function(){{drawBarLine('{cid}',d1,{{barLabel:'RPS',lineLabel:'RTFx',lineSuffix:'x',barDecimals:1}})}};requestAnimationFrame(w1._drawFn);_ro.observe(w1);"
            sweep_table = '<div class="table-wrap"><table><thead><tr><th class="text-c">C</th><th class="text-r">RPS</th><th class="text-r">RTFx</th><th class="text-r">p50</th><th class="text-r">p99</th><th class="text-c">Failures</th></tr></thead><tbody>'
            for l in levels:
                f = l.get("failures", 0)
                fc = "c-error" if f else "c-accent"
                sweep_table += f'<tr><td class="text-c">{l["concurrency"]}</td><td class="text-r">{l.get("rps","—")}</td><td class="text-r">{l.get("rtfx","—")}x</td><td class="text-r">{l.get("p50_ms","—")}ms</td><td class="text-r">{l.get("p99_ms","—")}ms</td><td class="text-c {fc}">{f}</td></tr>'
            sweep_table += '</tbody></table></div>'

    sustained_html = ""
    if sustained and sustained.get("rps", 0) > 0:
        sf = sustained.get("failures", 0)
        sfc = "c-error" if sf else "c-accent"
        sust_rtfx = sustained.get("rtfx", "—")
        sust_rps = sustained.get("rps", "—")
        sustained_html = f"""<div class="card"><div class="card-title">Sustained Load</div>
        <div class="metrics">
          <div class="metric"><div class="metric-val c-blue">{sustained.get("concurrency","—")}</div><div class="metric-label">Concurrency</div></div>
          <div class="metric"><div class="metric-val">{sust_rps}</div><div class="metric-label">RPS</div></div>
          <div class="metric"><div class="metric-val">{sust_rtfx}x</div><div class="metric-label">RTFx</div></div>
          <div class="metric"><div class="metric-val {sfc}">{sf}</div><div class="metric-label">Failures</div><div class="metric-note">{sustained.get("total_files","")} files, {sustained.get("rounds","")} rounds</div></div>
        </div></div>"""

    sys_html = ""
    sys_items = [
        ("GPU", gpu),
        ("GPU Memory", f'{sys_info.get("gpu_memory_mb",0)//1024} GB' if sys_info.get("gpu_memory_mb") else None),
        ("PyTorch", sys_info.get("pytorch_version")),
        ("CUDA", sys_info.get("cuda_version")),
        ("Driver", sys_info.get("driver_version")),
        ("Platform", sys_info.get("platform")),
        ("Container", sys_info.get("container_image")),
        ("Dataset", dataset),
        ("Samples", report.get("samples")),
        ("Normalization", norm),
    ]
    sys_items = [(l,v) for l,v in sys_items if v]
    if sys_items:
        sys_html = '<div class="card"><div class="card-title">System Info</div><div class="info-grid">'
        for label, val in sys_items:
            sys_html += f'<div class="info-item"><span class="info-label">{label}</span><span>{esc(str(val))}</span></div>'
        sys_html += '</div></div>'

    gates_html = ""
    gates = report.get("quality_gates", {}).get("gates", [])
    if gates:
        gates_html = '<div class="card"><div class="card-title">Quality Gates</div><div class="table-wrap"><table><thead><tr><th>Gate</th><th class="text-c">Result</th><th class="text-r">Threshold</th><th class="text-r">Actual</th></tr></thead><tbody>'
        for g in gates:
            gc = "c-accent" if g.get("passed") else "c-error"
            gates_html += f'<tr><td>{g.get("name","")}</td><td class="text-c {gc}">{"PASS" if g.get("passed") else "FAIL"}</td><td class="text-r">{g.get("threshold","—")}</td><td class="text-r">{g.get("actual","—")}</td></tr>'
        gates_html += '</tbody></table></div></div>'

    rps_label = "Peak Sess/min" if is_stream else "Peak RPS"
    rps_val = peak_sess if is_stream else peak_rps

    return assemble_page(run_id, badge, gpu, "", dataset, timestamp[:10] if timestamp else "",
        report.get("benchmark", ""),
        f"""<div class="metrics">
          <div class="metric"><div class="metric-val {"c-accent" if wer is not None and wer<4 else "c-blue"}">{fmt_num(wer,'%')}</div><div class="metric-label">WER</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(rps_val)}</div><div class="metric-label">{rps_label}</div></div>
          <div class="metric"><div class="metric-val">{fmt_num(peak_rtfx,'x')}</div><div class="metric-label">Peak RTFx</div></div>
          <div class="metric"><div class="metric-val {"c-accent" if total_failures==0 else "c-error"}">{total_failures}</div><div class="metric-label">Failures</div></div>
        </div>""",
        f'<div class="card"><div class="card-title">Concurrency Sweep</div>{chart_html}{sweep_table}</div>' + sustained_html,
        sys_html + gates_html,
        chart_init, git_sha=git_sha, timestamp=timestamp)


BUILDERS = {
    "2026-l4-batch-by-duration": build_duration_page,
    "2026-l4-dockerfile-stream": build_dockerfile_stream_page,
    "2026-l4-streaming-comparison": build_comparison_page,
}

def auto_detect_builder(run_id, result_dir):
    """Auto-detect the right builder based on report schema."""
    if run_id in BUILDERS:
        return BUILDERS[run_id]
    report = load_json(result_dir / "report.json")
    if report and report.get("schema_version") == "v1alpha2-live":
        return build_live_page
    return build_standard_page

generated = []
for d in sorted(RESULTS_DIR.iterdir()):
    if not d.is_dir(): continue
    run_id = d.name
    builder = auto_detect_builder(run_id, d)
    html = builder(run_id, d)
    if html:
        out = DOCS_DIR / f"{run_id}.html"
        out.write_text(html)
        generated.append(run_id)
        print(f"OK: {out.name}")

print(f"\n{len(generated)} pages generated in {DOCS_DIR}")

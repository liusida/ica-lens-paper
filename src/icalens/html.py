"""Write self-contained interactive ICA Lens HTML reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

# ruff: noqa: E501 -- Long lines keep the embedded HTML/JavaScript template readable.


# VS Code applies a larger visual scale to nested notebook webviews than JupyterLab.
# Users may override this module setting before displaying a result.
VSCODE_NOTEBOOK_SCALE = 0.6


def _gpt2_byte_hex(raw_token: str) -> str | None:
    """Recover original bytes from GPT-2's reversible byte-to-Unicode alphabet."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    characters = [chr(value) for value in byte_values]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            characters.append(chr(256 + extra))
            extra += 1
    decoder = dict(zip(characters, byte_values, strict=True))
    if not raw_token or any(character not in decoder for character in raw_token):
        return None
    return " ".join(f"{decoder[character]:02X}" for character in raw_token)


def _token_hint(item: dict[str, Any]) -> str:
    parts = []
    if item.get("token_id") is not None:
        parts.append(f"Token ID: {int(item['token_id'])}")
    raw_token = str(item.get("token", ""))
    if raw_token:
        parts.append(f"Raw token: {json.dumps(raw_token, ensure_ascii=False)}")
        byte_hex = _gpt2_byte_hex(raw_token)
        if byte_hex is not None:
            parts.append(f"Bytes: {byte_hex}")
    return " · ".join(parts)


def analysis_html(
    result: Any,
    *,
    metric: str = "score",
    top_k: int = 3,
    title: str = "ICA Lens Explorer",
) -> str:
    """Return a self-contained HTML document for an analysis result."""
    payload = _analysis_payload(result, metric=metric, top_k=top_k, title=title, compact=False)
    return _document(payload)


def analysis_iframe(
    result: Any,
    *,
    metric: str = "score",
    top_k: int = 3,
    title: str = "ICA Lens Explorer",
    height: int = 720,
) -> str:
    """Return an isolated interactive notebook representation."""
    if height <= 0:
        raise ValueError("height must be positive")
    payload = _analysis_payload(result, metric=metric, top_k=top_k, title=title, compact=True)
    document = _document(payload)
    source = html.escape(document, quote=True)
    return (
        '<iframe title="ICA Lens Explorer" sandbox="allow-scripts allow-same-origin" '
        'data-icalens-nested-frame="true" '
        f'srcdoc="{source}" style="width:100%;height:{height}px;border:0;overflow:hidden;" '
        'scrolling="no" '
        'loading="lazy"></iframe>'
    )


def write_analysis_html(
    result: Any,
    output_file: str | Path,
    *,
    metric: str = "score",
    top_k: int = 3,
    title: str = "ICA Lens Explorer",
) -> Path:
    """Write an interactive report from an :class:`AnalysisResult`."""
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        analysis_html(result, metric=metric, top_k=top_k, title=title),
        encoding="utf-8",
    )
    return destination


def component_profile_html(profile: Any, *, layer: int) -> str:
    """Return a standalone HTML document for one component profile."""
    return _component_profile_document(profile, layer=layer)


def component_profile_iframe(profile: Any, *, layer: int, height: int = 520) -> str:
    """Return an isolated notebook representation for one component profile."""
    if height <= 0:
        raise ValueError("height must be positive")
    source = html.escape(_component_profile_document(profile, layer=layer), quote=True)
    return (
        '<iframe title="ICA Lens Component Profile" sandbox="allow-scripts allow-same-origin" '
        'data-icalens-nested-frame="true" '
        f'srcdoc="{source}" style="width:100%;height:{height}px;border:0;overflow:hidden;" '
        'scrolling="no" loading="lazy"></iframe>'
    )


def write_component_profile_html(profile: Any, output_file: str | Path, *, layer: int) -> Path:
    """Write one component profile panel to a standalone HTML file."""
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(component_profile_html(profile, layer=layer), encoding="utf-8")
    return destination


def _visible_context(value: str) -> str:
    return value.replace("\r\n", "↵").replace("\r", "↵").replace("\n", "↵")


def _occurrence_target_span(item: dict[str, Any], context: str, target: str) -> tuple[int, int]:
    start = item.get("context_target_start")
    end = item.get("context_target_end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(context):
        return start, end
    if not target:
        return -1, -1
    matches: list[int] = []
    cursor = 0
    while True:
        index = context.find(target, cursor)
        if index < 0:
            break
        matches.append(index)
        cursor = index + 1
    if not matches:
        return -1, -1
    center = len(context) / 2
    closest = min(matches, key=lambda index: abs(index + len(target) / 2 - center))
    return closest, closest + len(target)


def _component_profile_document(profile: Any, *, layer: int) -> str:
    component = int(profile["component"])
    sign = str(profile["dominant_sign"])
    statistics = profile["sign_statistics"]
    score_statistics = profile.get("score_statistics") or {}
    examples = profile["examples"][sign]
    occurrences = examples.get("occurrences", [])
    logit_tokens = profile.get("logit_lens", {}).get("dominant", {}).get("top_tokens", [])[:10]
    r_lens_tokens = profile.get("r_lens", {}).get("dominant", {}).get("top_tokens", [])[:10]

    def percentage(value: Any) -> str:
        return f"{float(value) * 100:.1f}%"

    def token_chips(tokens: list[dict[str, Any]]) -> str:
        return "".join(
            f'<span class="profile-token-chip" title="{html.escape(_token_hint(item), quote=True)}"><span class="profile-value">'
            f"#{index}</span><strong>{html.escape(json.dumps(str(item.get('text', '')), ensure_ascii=False))}</strong></span>"
            for index, item in enumerate(tokens, start=1)
        )

    def highlighted_context(item: dict[str, Any]) -> str:
        target = str(item.get("text", ""))
        context = str(item.get("context", ""))
        start, end = _occurrence_target_span(item, context, target)
        if start < 0:
            return html.escape(_visible_context(context))
        return (
            html.escape(_visible_context(context[:start]))
            + '<span class="profile-target">'
            + html.escape(_visible_context(context[start:end]))
            + "</span>"
            + html.escape(_visible_context(context[end:]))
        )

    def occurrence_rank(item: dict[str, Any]) -> str:
        rank = item.get("absolute_score_rank")
        if not isinstance(rank, int) or rank < 1:
            return ""
        return (
            ' <span class="profile-occurrence-rank" '
            'title="Absolute-score rank among all components at this token">'
            f"(#{rank})</span>"
        )

    occurrence_items = "".join(
        "<li>"
        f'<span class="profile-occurrence-token" title="{html.escape(_token_hint(item), quote=True)}">{html.escape(json.dumps(str(item.get("text", "")), ensure_ascii=False))}</span>'
        f'<span class="profile-context">{highlighted_context(item)}</span>'
        '<span class="profile-occurrence-metrics">'
        f'<span title="Signed ICA score">↕ {float(item.get("score", 0)):+.2f}</span>{occurrence_rank(item)} · '
        f'<span title="Component energy">⚡ {percentage(item.get("energy", 0))}</span>'
        "</span></li>"
        for item in occurrences
    )
    r_lens = ""
    if r_lens_tokens:
        r_lens = (
            f'<div class="profile-token-row"><h3>R-lens tokens · {html.escape(sign)}</h3>'
            f'<div class="profile-chips">{token_chips(r_lens_tokens)}</div></div>'
        )

    position_positive = percentage(statistics["positive_fraction"])
    position_negative = percentage(statistics["negative_fraction"])
    energy_positive = percentage(statistics["positive_energy_fraction"])
    energy_negative = percentage(statistics["negative_energy_fraction"])
    excess_kurtosis = score_statistics.get("excess_kurtosis")
    excess_kurtosis_rank = score_statistics.get("excess_kurtosis_rank")
    skewness = score_statistics.get("skewness")
    fitting_statistics = profile.get("fitting_statistics") or {}
    kurtosis_html = (
        '<div class="profile-score-stat"><h3>Excess kurtosis</h3><div>'
        f'<strong class="profile-score-stat-value">{float(excess_kurtosis):.2f}</strong>'
        + (
            f'<span class="profile-score-stat-rank">rank #{int(excess_kurtosis_rank)}</span>'
            if excess_kurtosis_rank is not None
            else ""
        )
        + "</div></div>"
        if excess_kurtosis is not None
        else ""
    )
    logcosh_deviation = fitting_statistics.get("absolute_deviation")
    logcosh_rank = fitting_statistics.get("absolute_deviation_rank")
    logcosh_html = (
        '<div class="profile-score-stat"><h3 title="Absolute difference from the Gaussian '
        'baseline: |Logcosh contrast/objective − 0.374567|">Logcosh deviation</h3><div>'
        f'<strong class="profile-score-stat-value">{float(logcosh_deviation):.4f}</strong>'
        + (
            f'<span class="profile-score-stat-rank">rank #{int(logcosh_rank)}</span>'
            if logcosh_rank is not None
            else ""
        )
        + "</div></div>"
        if logcosh_deviation is not None
        else ""
    )
    tail_shape_html = f"{kurtosis_html}{logcosh_html}"
    score_statistics_html = (
        '<div class="profile-score-statistics">'
        '<div class="profile-score-stat"><h3>Skewness</h3>'
        f'<strong class="profile-score-stat-value">{float(skewness):+.2f}</strong></div>'
        f"{tail_shape_html}</div>"
        if skewness is not None and profile.get("tail_direction") is not None
        else (
            '<div class="profile-score-statistics"><div class="profile-sign-statistics"><h3>Sign distribution</h3>'
            f'<div class="profile-stat-row profile-stat-primary"><strong>Energy</strong><span>+{energy_positive}</span><span class="profile-bar"><span class="positive" style="width:{energy_positive}"></span><span class="negative"></span></span><span>−{energy_negative}</span></div>'
            f'<div class="profile-stat-row profile-stat-secondary"><strong>Positions</strong><span>+{position_positive}</span><span class="profile-bar"><span class="positive" style="width:{position_positive}"></span><span class="negative"></span></span><span>−{position_negative}</span></div>'
            f"</div>{tail_shape_html}</div>"
        )
    )
    direction_label = (
        f"{html.escape(sign)} tail"
        if profile.get("tail_direction") is not None
        else f"dominant {html.escape(sign)}"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ICA Lens component C{component}</title>
<style>
  :root {{--bg:#f6f7f9;--panel:#fff;--text:#151922;--muted:#647084;--border:#cbd3df}}
  * {{box-sizing:border-box}} body {{margin:0;padding:6px;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}}
  .panel {{padding:12px;background:var(--panel);border:1px solid var(--border);border-radius:8px}}
  summary {{cursor:pointer;font-weight:800}} .profile-grid {{display:grid;grid-template-columns:minmax(0,1fr) 150px;gap:16px 24px;margin-top:12px}}
  h3 {{margin:0 0 7px;font-size:13px}} .profile-wide {{grid-column:1/-1;padding-top:12px;border-top:1px solid #e1e6ee}}
  .profile-token-row + .profile-token-row {{margin-top:13px}} .profile-stat-row {{display:grid;grid-template-columns:62px 48px minmax(80px,1fr) 48px;gap:7px;align-items:center;margin:7px 0;color:#435066;font-size:11px}}
  .profile-stat-primary {{margin-bottom:10px;color:#273244;font-size:12px}} .profile-stat-secondary {{opacity:.65}} .profile-bar {{display:flex;height:9px;overflow:hidden;border-radius:999px;background:#e6e9ef}}
  .profile-stat-primary .profile-bar {{height:14px}} .positive {{background:#f59e0b}} .negative {{flex:1;background:#1e3a8a}}
  .profile-score-statistics {{display:flex;flex-direction:column;gap:12px;font-variant-numeric:tabular-nums}}
  .profile-score-stat {{min-width:0}} .profile-score-stat h3 {{margin-bottom:2px;font-size:11px}}
  .profile-score-stat-value {{color:#273244;font-size:15px}}
  .profile-score-stat-rank {{margin-left:6px;color:#647084;font-size:10px}}
  .profile-chips {{display:flex;flex-wrap:wrap;gap:6px}} .profile-token-chip {{display:inline-flex;gap:6px;padding:4px 7px;border:1px solid #d5dce7;border-radius:999px;background:#f8fafc;font-size:11px}}
  .profile-value {{color:var(--muted);font-variant-numeric:tabular-nums}} ol {{margin:0;padding-left:20px;columns:3 360px;column-gap:32px}} li {{break-inside:avoid;margin:0 0 8px;overflow-wrap:anywhere}}
  .profile-occurrence-token {{color:#435066;font-size:11px}} .profile-context {{display:block;color:#3f4a5c;font-size:13px;line-height:1.5}} .profile-occurrence-metrics {{display:block;margin-top:2px;color:#929bab;font-size:9px}} .profile-occurrence-rank {{font-size:8px;color:#7f8999}}
  .profile-target {{padding:0 1px;border-radius:2px;background:#fff1a8;color:#273244;text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:2px}}
  @media(max-width:700px){{.profile-grid{{grid-template-columns:1fr}}.profile-wide{{grid-column:auto}}}}
</style></head><body>
<details class="panel" open><summary>Component profile — C{component} · layer {int(layer)} · {direction_label}</summary>
<div class="profile-grid">
  <section><div class="profile-token-row"><h3>Logit-lens tokens · {html.escape(sign)}</h3><div class="profile-chips">{token_chips(logit_tokens)}</div></div>{r_lens}</section>
  <section>{score_statistics_html}</section>
  <section class="profile-wide"><h3>Top-score occurrences · {html.escape(sign)}</h3><ol>{occurrence_items}</ol></section>
</div></details>
<script>
const vscodeHost=(()=>{{
  if(/Electron|Code[/]/.test(navigator.userAgent))return true;
  try{{
    const parentDocument=window.parent.document;
    const parentStyle=window.parent.getComputedStyle(parentDocument.body||parentDocument.documentElement);
    return /vscode-webview|vscode-resource/.test(window.parent.location.href)||
      Boolean(parentStyle.getPropertyValue("--vscode-font-family").trim())||
      Boolean(parentDocument.querySelector(".vscode-body"));
  }}catch(_error){{return false}}
}})();
const isNestedFrame=()=>window.frameElement?.dataset.icalensNestedFrame==="true";
const notebookScale=()=>isNestedFrame()&&vscodeHost?{VSCODE_NOTEBOOK_SCALE}:1;
const applyVSCodeScale=()=>{{
  if(isNestedFrame()&&vscodeHost)document.documentElement.style.zoom="{VSCODE_NOTEBOOK_SCALE}";
}};
const resize=()=>{{
  if(!isNestedFrame())return;
  const panel=document.querySelector(".panel");
  const intrinsicHeight=panel?panel.scrollHeight+12:document.body.scrollHeight;
  window.frameElement.style.height=Math.max(Math.ceil(intrinsicHeight*notebookScale())+4,180)+"px";
}};
applyVSCodeScale();addEventListener("load",resize);new ResizeObserver(resize).observe(document.body);
</script>
</body></html>"""


def _analysis_payload(
    result: Any,
    *,
    metric: str,
    top_k: int,
    title: str,
    compact: bool,
) -> str:
    """Serialize the display portion of an analysis result."""
    if metric not in {"score", "energy"}:
        raise ValueError("metric must be 'score' or 'energy'")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores = result.scores.detach().float().cpu()
    energy = result.energy.detach().float().cpu()
    component_count = int(scores.shape[-1])
    tokens = [
        {
            "token_index": row,
            "position": int(result.positions[row]),
            "token": result.tokens[row],
            "token_text": result.token_texts[row],
            "token_label": result.token_labels[row],
            "token_tooltip": result.token_tooltips[row],
            "metrics": {
                "score": scores[row].tolist(),
                "energy": energy[row].tolist(),
            },
        }
        for row in range(len(result.tokens))
    ]
    token_groups: list[dict[str, Any]] = []
    if result.token_groups:
        for title in dict.fromkeys(result.token_groups):
            grouped = [
                token
                for token, group in zip(tokens, result.token_groups, strict=True)
                if group == title
            ]
            if grouped:
                token_groups.append({"title": title, "open": True, "tokens": grouped})
        tokens = []
    return _payload(
        title=title,
        model=result.model,
        layer=result.layer,
        input_text=result.input_text,
        token_scope=result.token_scope,
        metric=metric,
        top_k=min(top_k, component_count),
        component_count=component_count,
        dynamic_metrics=True,
        compact=compact,
        result_group_title="Tokens",
        messages=list(result.messages),
        tokens=tokens,
        token_groups=token_groups,
        selected_components=list(result.selected_components),
        component_profiles=result.component_profiles or {},
        logit_effects=result.logit_effects,
    )


def write_explorer_html(
    output_file: str | Path,
    *,
    title: str,
    model: str,
    layer: int,
    input_text: str,
    token_scope: str,
    tokens: list[dict[str, Any]],
    token_groups: list[dict[str, Any]] | None = None,
    metric: str = "score",
    result_group_title: str = "Tokens",
    messages: list[dict[str, str]] | None = None,
) -> Path:
    """Write an interactive standalone HTML report and return its absolute path."""
    if metric not in {"score", "energy"}:
        raise ValueError("metric must be 'score' or 'energy'")
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(
        title=title,
        model=model,
        layer=layer,
        input_text=input_text,
        token_scope=token_scope,
        metric=metric,
        top_k=max((len(token.get("top", [])) for token in tokens), default=0),
        component_count=0,
        dynamic_metrics=False,
        compact=False,
        result_group_title=result_group_title,
        messages=messages or [],
        tokens=tokens,
        token_groups=token_groups or [],
        component_profiles={},
        logit_effects={},
    )
    destination.write_text(_document(payload), encoding="utf-8")
    return destination


def _payload(**values: Any) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _document(payload: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ICA Lens Explorer</title>
  <style>
    :root {{
      --bg: #f6f7f9; --panel: #fff; --text: #151922; --muted: #647084;
      --border: #cbd3df; --accent: #1f6feb; --shadow: 0 1px 2px rgb(20 25 34/.08),
        0 10px 30px rgb(20 25 34/.06); --card-width: 100px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 12px 18px; background: var(--panel);
      border-bottom: 1px solid var(--border); box-shadow: var(--shadow); }}
    header h1 {{ margin: 0; font-size: 19px; }}
    main {{ max-width: 1680px; margin: 0 auto; padding: 18px; }}
    body.compact header {{ display: none; }}
    body.compact {{ overflow: hidden; }}
    body.compact main {{ padding: 6px; }}
    body.compact .panel {{ margin-bottom: 8px; box-shadow: none; }}
    .panel {{ margin-bottom: 14px; padding: 12px; background: var(--panel);
      border: 1px solid var(--border); border-radius: 8px; box-shadow: var(--shadow); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 7px; color: var(--muted); }}
    .analysis-context > summary {{ color: var(--muted); font-size: 12px; }}
    .analysis-context[open] > summary {{ margin-bottom: 9px; }}
    .chip {{ padding: 3px 7px; border: 1px solid var(--border); border-radius: 999px;
      background: #f8fafc; }}
    .chip-label {{ color: #445066; font-weight: 750; }}
    .context-label {{ margin-top: 9px; color: #445066; font-size: 11px;
      font-weight: 750; }}
    .input {{ margin: 4px 0 0; padding: 1px 0 1px 10px; overflow-wrap: anywhere;
      white-space: pre-wrap; border-left: 2px solid #c5cfdd; color: #273244;
      font: 500 13px/1.65 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      letter-spacing: .01em; }}
    .conversation {{ display: grid; gap: 9px; margin-top: 12px; }}
    .message {{ width: min(88%, 1050px); padding: 9px 11px; border: 1px solid #d8e0eb;
      border-radius: 9px; background: #f8fafc; }}
    .message.user {{ margin-left: auto; background: #eef5ff; border-color: #bfd3f5; }}
    .message.assistant {{ margin-right: auto; background: #f7f8fa; }}
    .message.system {{ width: 100%; background: #fff8e8; border-color: #ead7a4; }}
    .message-role {{ margin-bottom: 3px; color: var(--muted); font-size: 11px;
      font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }}
    .message-content {{ overflow-wrap: anywhere; white-space: pre-wrap; }}
    details > summary {{ cursor: pointer; font-weight: 800; }}
    details .results {{ margin-top: 12px; }}
    .controls {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; }}
    label {{ display: flex; align-items: center; gap: 7px; color: #435066;
      font-size: 12px; font-weight: 700; }}
    input, select {{ width: 76px; padding: 5px 7px; border: 1px solid var(--border);
      border-radius: 6px; background: #fff; }}
    button {{ padding: 5px 9px; border: 1px solid var(--border); border-radius: 6px;
      background: #fff; cursor: pointer; font: inherit; font-weight: 700; }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    .selection {{ margin-left: auto; color: var(--muted); font-size: 12px; font-weight: 700; }}
    .results {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--card-width), 1fr));
      gap: 10px; }}
    .token-card {{ min-width: 0; padding: 6px; text-align: center; background: #fff;
      border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 1px 2px rgb(20 25 34/.04); }}
    .token-text {{ min-height: 25px; margin: 1px 0 6px; overflow: hidden;
      white-space: nowrap; text-overflow: ellipsis; font-weight: 850; }}
    .score-row {{ position: relative; min-height: 24px; margin-top: 3px; }}
    .badge {{ position: relative; display: flex; width: 100%; min-height: 24px; align-items: center;
      justify-content: space-between; gap: 5px; padding: 3px 5px; overflow: hidden;
      border: 1px solid #d5dce7; border-radius: 5px; color: var(--text); font-size: 10px;
      background: linear-gradient(90deg, var(--bar) 0, var(--bar) var(--width),
        transparent var(--width), transparent 100%); }}
    .badge.weak {{ opacity: .35; }}
    .badge.selected {{ outline: 2px solid var(--color); border-color: var(--color); opacity: 1; }}
    .component {{ font-weight: 850; }}
    .score {{ font-variant-numeric: tabular-nums; }}
    .profile-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 150px;
      gap: 16px 24px; margin-top: 12px; }}
    .profile-wide {{ grid-column: 1 / -1; padding-top: 12px; border-top: 1px solid #e1e6ee; }}
    .profile-section h3 {{ margin: 0 0 7px; font-size: 13px; }}
    .profile-token-row + .profile-token-row {{ margin-top: 13px; }}
    .profile-stat-row {{ display: grid; grid-template-columns: 62px 48px minmax(100px, 1fr) 48px;
      gap: 7px; align-items: center; margin: 7px 0; color: #435066; font-size: 11px; }}
    .profile-stat-row strong:last-child {{ text-align: right; }}
    .profile-bar {{ display: flex; height: 9px; overflow: hidden; border-radius: 999px;
      background: #e6e9ef; }}
    .profile-stat-primary {{ margin-bottom: 10px; color: #273244; font-size: 12px; }}
    .profile-stat-primary .profile-bar {{ height: 14px; }}
    .profile-stat-secondary {{ opacity: .65; }}
    .profile-score-statistics {{ display: flex; flex-direction: column; gap: 12px;
      font-variant-numeric: tabular-nums; }}
    .profile-score-stat {{ min-width: 0; }}
    .profile-score-stat h3 {{ margin-bottom: 2px; font-size: 11px; }}
    .profile-score-stat-value {{ color: #273244; font-size: 15px; }}
    .profile-score-stat-rank {{ margin-left: 6px; color: #647084; font-size: 10px; }}
    .profile-bar-positive {{ background: #f59e0b; }}
    .profile-bar-negative {{ flex: 1; background: #1e3a8a; }}
    .profile-list {{ margin: 0; padding-left: 20px; }}
    .profile-list li {{ margin: 5px 0; overflow-wrap: anywhere; }}
    .profile-occurrences {{ columns: 3 360px; column-gap: 32px; }}
    .profile-occurrences li {{ break-inside: avoid; margin: 0 0 8px; }}
    .profile-chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .profile-token-chip {{ display: inline-flex; gap: 6px; padding: 4px 7px;
      border: 1px solid #d5dce7; border-radius: 999px; background: #f8fafc; font-size: 11px; }}
    .profile-value {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .profile-occurrence-token {{ color: #435066; font-size: 11px; }}
    .profile-occurrence-metrics {{ display: block; margin-top: 2px; color: #929bab;
      font-size: 9px; font-weight: 400; }}
    .profile-occurrence-rank {{ color: #7f8999; font-size: 8px; }}
    .profile-metric {{ white-space: nowrap; }}
    .profile-metric-icon {{ font-weight: 600; }}
    .profile-context {{ display: block; color: #3f4a5c; font-size: 13px; line-height: 1.5; }}
    .profile-target {{ padding: 0 1px; border-radius: 2px; background: #fff1a8;
      color: #273244; font-weight: inherit; text-decoration: underline;
      text-decoration-thickness: 2px; text-underline-offset: 2px; }}
    @media (max-width: 700px) {{ main {{ padding: 10px; }} .selection {{ flex-basis: 100%; }}
      .profile-grid {{ grid-template-columns: 1fr; }} .profile-wide {{ grid-column: auto; }} }}
  </style>
</head>
<body>
  <header><h1 id="title">ICA Lens Explorer</h1></header>
  <main>
    <details class="panel analysis-context" id="analysisContext">
      <summary>Analysis details</summary>
      <div class="meta">
        <span class="chip" id="model"></span><span class="chip" id="layer"></span>
        <span class="chip" id="scope"></span><span class="chip" id="metric"></span>
        <span class="chip" id="turns" hidden></span><span class="chip" id="count"></span>
      </div>
      <div class="context-label" id="inputLabel">Analyzed text</div>
      <blockquote class="input" id="input"></blockquote>
      <div class="conversation" id="conversation" hidden></div>
    </details>
    <section class="panel controls">
      <label id="metricControl">Metric <select id="metricSelect"><option value="score">Score</option><option value="energy">Energy</option></select></label>
      <label id="topKControl">Top K <input id="topK" type="number" min="1" step="1" value="3"></label>
      <label>Card width <input id="cardWidth" type="number" min="60" max="360" step="20" value="100"></label>
      <label>Opacity cutoff <input id="cutoff" type="number" min="0" max="1" step="0.05" value="0.5"></label>
      <button id="clear" type="button">Clear selection</button>
      <span class="selection" id="selection">Click a component to highlight it.</span>
    </section>
    <div id="tokenGroups"></div>
    <details class="panel" id="resultGroup" open>
      <summary id="resultSummary"></summary>
      <section class="results" id="results"></section>
    </details>
    <details class="panel" id="componentProfile" hidden>
      <summary id="componentProfileSummary">Component profile</summary>
      <div class="profile-grid" id="componentProfileBody"></div>
    </details>
    <details class="panel" id="localIntervention" hidden>
      <summary id="localInterventionSummary">Local intervention projection</summary>
      <div class="profile-chips" id="localInterventionBody"></div>
    </details>
  </main>
  <script>
    const data = {payload};
    const vscodeHost = (() => {{
      if (/Electron|Code[/]/.test(navigator.userAgent)) return true;
      try {{
        const parentDocument = window.parent.document;
        const parentStyle = window.parent.getComputedStyle(
          parentDocument.body || parentDocument.documentElement
        );
        return /vscode-webview|vscode-resource/.test(window.parent.location.href) ||
          Boolean(parentStyle.getPropertyValue("--vscode-font-family").trim()) ||
          Boolean(parentDocument.querySelector(".vscode-body"));
      }} catch (_error) {{
        return false;
      }}
    }})();
    const esc = value => String(value).replace(/[&<>"']/g, char => ({{
      "&":"&amp;", "<":"&lt;", ">":"&gt;", "\\\"":"&quot;", "'":"&#39;"
    }})[char]);
    const gpt2ByteDecoder = (() => {{
      const bytes = [];
      for (let value = 33; value <= 126; value++) bytes.push(value);
      for (let value = 161; value <= 172; value++) bytes.push(value);
      for (let value = 174; value <= 255; value++) bytes.push(value);
      const characters = bytes.map(value => String.fromCodePoint(value));
      let extra = 0;
      for (let value = 0; value < 256; value++) {{
        if (!bytes.includes(value)) {{
          bytes.push(value);
          characters.push(String.fromCodePoint(256 + extra));
          extra += 1;
        }}
      }}
      return Object.fromEntries(characters.map((character, index) => [character, bytes[index]]));
    }})();
    const tokenHint = item => {{
      const parts = [];
      if (item.token_id !== undefined && item.token_id !== null) parts.push(`Token ID: ${{item.token_id}}`);
      const raw = String(item.token || "");
      if (raw) {{
        parts.push(`Raw token: ${{JSON.stringify(raw)}}`);
        const bytes = [...raw].map(character => gpt2ByteDecoder[character]);
        if (bytes.every(value => value !== undefined)) {{
          parts.push(`Bytes: ${{bytes.map(value => value.toString(16).toUpperCase().padStart(2, "0")).join(" ")}}`);
        }}
      }}
      return parts.join(" · ");
    }};
    const color = component => `hsl(${{(Number(component) * 137.508) % 360}} 66% 78%)`;
    const state = {{ selected: data.selected_components?.[0] ?? null, selectedToken: null, metric: data.metric, topK: Number(data.top_k || 3) }};
    const results = document.getElementById("results");
    const cutoff = document.getElementById("cutoff");

    document.body.classList.toggle("compact", Boolean(data.compact));
    const isNestedFrame = () => window.frameElement?.dataset.icalensNestedFrame === "true";
    const notebookScale = data.compact && isNestedFrame() && vscodeHost
      ? {VSCODE_NOTEBOOK_SCALE}
      : 1;
    if (data.compact && isNestedFrame() && vscodeHost) {{
      document.documentElement.style.zoom = "{VSCODE_NOTEBOOK_SCALE}";
    }}
    document.getElementById("analysisContext").open = !data.compact;
    if (data.compact) {{
      document.querySelector("main").append(document.getElementById("analysisContext"));
    }}
    document.getElementById("title").textContent = data.title;
    const modelChip = document.getElementById("model");
    const revisionAt = data.model.lastIndexOf("@");
    const shortModel = revisionAt < 0
      ? data.model
      : `${{data.model.slice(0, revisionAt)}}@${{data.model.slice(revisionAt + 1, revisionAt + 9)}}`;
    modelChip.innerHTML = `<span class="chip-label">Model:</span> ${{esc(shortModel)}}`;
    modelChip.title = data.model;
    document.getElementById("layer").innerHTML = `<span class="chip-label">Layer:</span> ${{data.layer}}`;
    const scopeLabels = {{
      "all text tokens": "entire input",
      "all": "entire conversation",
      "content": "message content",
      "user": "user messages",
      "assistant": "assistant messages"
    }};
    const scopeLabel = scopeLabels[data.token_scope] || data.token_scope;
    document.getElementById("scope").innerHTML = `<span class="chip-label">Token scope:</span> ${{esc(scopeLabel)}}`;
    document.getElementById("metric").innerHTML = `<span class="chip-label">Metric:</span> ${{esc(state.metric)}}`;
    const metricSelect = document.getElementById("metricSelect");
    const topK = document.getElementById("topK");
    metricSelect.value = state.metric;
    topK.value = state.topK;
    topK.max = Math.max(1, Number(data.component_count || state.topK));
    document.getElementById("metricControl").hidden = !data.dynamic_metrics;
    document.getElementById("topKControl").hidden = !data.dynamic_metrics;
    const totalTokens = data.tokens.length + data.token_groups.reduce(
      (total, group) => total + group.tokens.length, 0
    );
    document.getElementById("count").innerHTML = `<span class="chip-label">Tokens:</span> ${{totalTokens}}`;
    document.getElementById("input").textContent = data.input_text;
    if (data.messages.length) {{
      const input = document.getElementById("input");
      document.getElementById("inputLabel").textContent = "Analyzed conversation";
      const conversation = document.getElementById("conversation");
      const turns = document.getElementById("turns");
      input.hidden = true;
      conversation.hidden = false;
      const roleCounts = {{ user: 0, assistant: 0, system: 0 }};
      data.messages.forEach(message => {{
        const role = String(message.role);
        roleCounts[role] = (roleCounts[role] || 0) + 1;
        const card = document.createElement("article");
        card.className = `message ${{role}}`;
        const label = document.createElement("div");
        label.className = "message-role";
        label.textContent = role === "system"
          ? "System"
          : `${{role === "user" ? "User" : "Assistant"}} ${{roleCounts[role]}}`;
        const content = document.createElement("div");
        content.className = "message-content";
        content.textContent = message.content;
        card.append(label, content);
        conversation.append(card);
      }});
      turns.hidden = false;
      turns.innerHTML = `<span class="chip-label">User turns:</span> ${{roleCounts.user}}`;
    }}
    document.getElementById("resultSummary").textContent =
      `${{data.result_group_title}} (${{data.tokens.length}} tokens)`;
    document.getElementById("resultGroup").hidden = data.tokens.length === 0;
    const tokenGroups = document.getElementById("tokenGroups");
    const groupResults = [];
    data.token_groups.forEach(group => {{
      const details = document.createElement("details");
      details.className = "panel";
      details.open = Boolean(group.open);
      const summary = document.createElement("summary");
      summary.textContent = `${{group.title}} (${{group.tokens.length}} tokens)`;
      const cards = document.createElement("section");
      cards.className = "results";
      details.append(summary, cards);
      tokenGroups.append(details);
      groupResults.push({{ cards, tokens: group.tokens }});
    }});

    function tokenItems(token) {{
      if (!token.metrics) return token.top;
      const values = token.metrics[state.metric];
      return values.map((score, component) => ({{ component, score: Number(score) }}))
        .sort((left, right) => state.metric === "score"
          ? Math.abs(right.score) - Math.abs(left.score)
          : right.score - left.score)
        .slice(0, state.topK);
    }}

    function tokenCard(token, threshold) {{
      const tokenText = String(token.token_label || token.token_text || token.token).replace(/\\r\\n|\\r|\\n/g, "↵");
      const tokenTooltip = String(token.token_tooltip || token.token);
      const items = tokenItems(token);
      const peak = Math.max(0, ...items.map(item => Math.abs(Number(item.score))));
      const badges = items.map(item => {{
        const component = Number(item.component);
        const score = Number(item.score);
        const ratio = peak ? Math.abs(score) / peak : 0;
        const selected = state.selected === component;
        const bar = selected ? color(component) : "#e7ebf1";
        const display = state.metric === "energy"
          ? `${{(score * 100).toFixed(2)}}%`
          : `${{score >= 0 ? "+" : ""}}${{score.toFixed(3)}}`;
        const tooltip = componentTooltip(component);
        return `<div class="score-row"><button class="badge ${{ratio < threshold ? "weak" : ""}} ${{selected ? "selected" : ""}}" data-component="${{component}}" data-token-index="${{token.token_index}}" title="${{esc(tooltip)}}" style="--width:${{(ratio*100).toFixed(1)}}%;--bar:${{bar}};--color:${{color(component)}}"><span class="component">C${{component}}</span><span class="score">${{display}}</span></button></div>`;
      }}).join("");
      return `<article class="token-card"><div class="token-text" title="${{esc(tokenTooltip)}}">${{esc(tokenText)}}</div>${{badges}}</article>`;
    }}

    function componentTooltip(component) {{
      const profile = data.component_profiles[String(component)] || data.component_profiles[component];
      if (!profile) return `C${{component}} · no component profile available`;
      const direction = profile.tail_direction
        ? `${{profile.tail_direction}} tail`
        : `dominant ${{profile.dominant_sign}}`;
      const lines = [`C${{component}} · ${{direction}}`];
      lines.push("Top occurrences:");
      profile.occurrences.slice(0, 3).forEach((item, index) => {{
        const text = String(item.text || "").replace(/\\r\\n|\\r|\\n/g, "↵");
        const context = String(item.context || "").replace(/\\r\\n|\\r|\\n/g, "↵");
        lines.push(`${{index + 1}}. ${{JSON.stringify(text)}} · ${{(Number(item.energy) * 100).toFixed(1)}}% · ${{context}}`);
      }});
      lines.push("Top logit-lens tokens:");
      profile.logit_tokens.slice(0, 3).forEach((item, index) => {{
        lines.push(`#${{index + 1}} ${{JSON.stringify(item.text)}}`);
      }});
      if (profile.r_lens_tokens?.length) {{
        lines.push("Top R-lens tokens:");
        profile.r_lens_tokens.slice(0, 3).forEach((item, index) => {{
          lines.push(`#${{index + 1}} ${{JSON.stringify(item.text)}}`);
        }});
      }}
      return lines.join("\\n");
    }}

    function allTokens() {{
      return [...data.tokens, ...data.token_groups.flatMap(group => group.tokens)];
    }}

    function updateSelection() {{
      const selection = document.getElementById("selection");
      const profilePanel = document.getElementById("componentProfile");
      const localPanel = document.getElementById("localIntervention");
      if (state.selected === null) {{
        selection.textContent = "Click a component to highlight it.";
        profilePanel.hidden = true;
        localPanel.hidden = true;
        return;
      }}
      const tokens = allTokens();
      const observations = tokens.map(token => {{
        const raw = token.metrics ? token.metrics[state.metric][state.selected] : null;
        const listed = tokenItems(token).find(item => Number(item.component) === state.selected);
        return {{ token, value: raw === null ? Number(listed?.score) : Number(raw), visible: Boolean(listed) }};
      }}).filter(item => Number.isFinite(item.value));
      const strongest = observations.reduce((best, item) =>
        best === null || Math.abs(item.value) > Math.abs(best.value) ? item : best, null);
      const visible = observations.filter(item => item.visible).length;
      const value = strongest === null ? "unavailable" : state.metric === "energy"
        ? `${{(strongest.value * 100).toFixed(2)}}%`
        : `${{strongest.value >= 0 ? "+" : ""}}${{strongest.value.toFixed(3)}}`;
      const token = strongest === null ? "" : ` at “${{strongest.token.token_text || strongest.token.token}}” (${{value}})`;
      selection.textContent = `C${{state.selected}} · strongest${{token}} · visible in ${{visible}}/${{tokens.length}} tokens`;
      updateComponentProfile(state.selected, state.selectedToken);
      updateLocalIntervention(state.selected, state.selectedToken);
    }}

    function updateComponentProfile(component, tokenIndex) {{
      const panel = document.getElementById("componentProfile");
      const summary = document.getElementById("componentProfileSummary");
      const body = document.getElementById("componentProfileBody");
      const profile = data.component_profiles[String(component)] || data.component_profiles[component];
      panel.hidden = false;
      if (!profile) {{
        summary.textContent = `Component profile — C${{component}}`;
        body.innerHTML = '<div class="profile-section">No component profile is available.</div>';
        return;
      }}
      const direction = profile.tail_direction
        ? `${{profile.tail_direction}} tail`
        : `dominant ${{profile.dominant_sign}}`;
      summary.textContent = `Component profile — C${{component}} · ${{direction}}`;
      const stats = profile.sign_statistics;
      const percentage = value => `${{(Number(value) * 100).toFixed(1)}}%`;
      const highlightedContext = item => {{
        const target = String(item.text || "");
        const context = String(item.context || "");
        const visible = value => value.replace(/\\r\\n|\\r|\\n/g, "↵");
        let start = Number.isInteger(item.context_target_start)
          ? item.context_target_start : -1;
        let end = Number.isInteger(item.context_target_end)
          ? item.context_target_end : -1;
        if (!(start >= 0 && end > start && end <= context.length)) {{
          const matches = [];
          if (target) {{
            let cursor = 0;
            while (cursor <= context.length - target.length) {{
              const index = context.indexOf(target, cursor);
              if (index < 0) break;
              matches.push(index);
              cursor = index + 1;
            }}
          }}
          if (!matches.length) return esc(visible(context));
          const center = context.length / 2;
          start = matches.reduce((best, index) =>
            Math.abs(index + target.length / 2 - center) <
              Math.abs(best + target.length / 2 - center) ? index : best
          );
          end = start + target.length;
        }}
        return esc(visible(context.slice(0, start))) +
          `<span class="profile-target">${{esc(visible(context.slice(start, end)))}}</span>` +
          esc(visible(context.slice(end)));
      }};
      const occurrenceItems = profile.occurrences.map(item =>
        `<li><span class="profile-occurrence-token" title="${{esc(tokenHint(item))}}">${{esc(JSON.stringify(String(item.text || "")))}}</span>` +
        `<span class="profile-context">${{highlightedContext(item)}}</span>` +
        `<span class="profile-occurrence-metrics"><span class="profile-metric" title="Signed ICA score" aria-label="Signed ICA score"><span class="profile-metric-icon">↕</span> ${{Number(item.score) >= 0 ? "+" : ""}}${{Number(item.score).toFixed(2)}}</span>` +
        `${{Number.isInteger(item.absolute_score_rank) && item.absolute_score_rank > 0 ? ` <span class="profile-occurrence-rank" title="Absolute-score rank among all components at this token">(#${{item.absolute_score_rank}})</span>` : ""}} · ` +
        `<span class="profile-metric" title="Component energy" aria-label="Component energy"><span class="profile-metric-icon">⚡</span> ${{percentage(item.energy)}}</span></span></li>`
      ).join("");
      const tokenItems = items => items.map((item, index) =>
        `<span class="profile-token-chip" title="${{esc(tokenHint(item))}}"><span class="profile-value">#${{index + 1}}</span>` +
        `<strong>${{esc(JSON.stringify(String(item.text || "")))}}</strong></span>`
      ).join("");
      const positionPositive = percentage(stats.positive_fraction);
      const positionNegative = percentage(stats.negative_fraction);
      const energyPositive = percentage(stats.positive_energy_fraction);
      const energyNegative = percentage(stats.negative_energy_fraction);
      const excessKurtosis = Number(profile.score_statistics?.excess_kurtosis);
      const excessKurtosisRank = Number(profile.score_statistics?.excess_kurtosis_rank);
      const skewness = Number(profile.score_statistics?.skewness);
      const logcoshDeviation = Number(profile.fitting_statistics?.absolute_deviation);
      const logcoshRank = Number(profile.fitting_statistics?.absolute_deviation_rank);
      const kurtosisRow = Number.isFinite(excessKurtosis)
        ? `<div class="profile-score-stat"><h3>Excess kurtosis</h3><div><strong class="profile-score-stat-value">${{excessKurtosis.toFixed(2)}}</strong>${{Number.isInteger(excessKurtosisRank) ? `<span class="profile-score-stat-rank">rank #${{excessKurtosisRank}} / ${{data.component_count}}</span>` : ""}}</div></div>`
        : "";
      const logcoshRow = Number.isFinite(logcoshDeviation)
        ? `<div class="profile-score-stat"><h3 title="Absolute difference from the Gaussian baseline: |Logcosh contrast/objective − 0.374567|">Logcosh deviation</h3><div><strong class="profile-score-stat-value">${{logcoshDeviation.toFixed(4)}}</strong>${{Number.isInteger(logcoshRank) ? `<span class="profile-score-stat-rank">rank #${{logcoshRank}} / ${{data.component_count}}</span>` : ""}}</div></div>`
        : "";
      const tailShapeRows = `${{kurtosisRow}}${{logcoshRow}}`;
      const scoreStatistics = profile.tail_direction && Number.isFinite(skewness)
        ? `<div class="profile-score-statistics"><div class="profile-score-stat"><h3>Skewness</h3><strong class="profile-score-stat-value">${{skewness >= 0 ? "+" : ""}}${{skewness.toFixed(2)}}</strong></div>${{tailShapeRows}}</div>`
        : `<div class="profile-score-statistics"><div class="profile-sign-statistics"><h3>Sign distribution</h3>
          <div class="profile-stat-row profile-stat-primary"><strong>Energy</strong><span>+${{energyPositive}}</span><span class="profile-bar"><span class="profile-bar-positive" style="width:${{energyPositive}}"></span><span class="profile-bar-negative"></span></span><span>−${{energyNegative}}</span></div>
          <div class="profile-stat-row profile-stat-secondary"><strong>Positions</strong><span>+${{positionPositive}}</span><span class="profile-bar"><span class="profile-bar-positive" style="width:${{positionPositive}}"></span><span class="profile-bar-negative"></span></span><span>−${{positionNegative}}</span></div></div>${{tailShapeRows}}</div>`;
      const rLensSection = profile.r_lens_tokens?.length
        ? `<div class="profile-token-row"><h3>R-lens tokens · ${{profile.dominant_sign}}</h3><div class="profile-chips">${{tokenItems(profile.r_lens_tokens)}}</div></div>`
        : "";
      body.innerHTML = `
        <section class="profile-section">
          <div class="profile-token-row"><h3>Logit-lens tokens · ${{profile.dominant_sign}}</h3><div class="profile-chips">${{tokenItems(profile.logit_tokens)}}</div></div>
          ${{rLensSection}}
        </section>
        <section class="profile-section">${{scoreStatistics}}</section>
        <section class="profile-section profile-wide"><h3>Top-score occurrences · ${{profile.dominant_sign}}</h3><ol class="profile-list profile-occurrences">${{occurrenceItems}}</ol></section>`;
    }}

    function updateLocalIntervention(component, tokenIndex) {{
      const panel = document.getElementById("localIntervention");
      const summary = document.getElementById("localInterventionSummary");
      const body = document.getElementById("localInterventionBody");
      const effect = data.logit_effects.find(item =>
        Number(item.component) === component && Number(item.token_index) === tokenIndex);
      if (!effect) {{
        panel.hidden = true;
        return;
      }}
      panel.hidden = false;
      summary.textContent = `Local intervention projection — C${{component}} ×${{Number(effect.multiplier).toFixed(2)}}`;
      body.innerHTML = effect.tokens.map((item, index) =>
        `<span class="profile-token-chip" title="Ranked by absolute local logit change"><span class="profile-value">#${{index + 1}}</span>` +
        `<strong>${{esc(JSON.stringify(String(item.text || "")))}}</strong></span>`
      ).join("");
    }}

    function resizeFrame() {{
      if (!data.compact || !isNestedFrame()) return;
      const main = document.querySelector("main");
      const contentHeight = main ? Math.ceil(main.scrollHeight * notebookScale) : 0;
      const height = Math.max(contentHeight + 4, 180);
      window.frameElement.style.height = `${{height}}px`;
    }}

    function render() {{
      const threshold = Math.max(0, Math.min(1, Number(cutoff.value || .5)));
      results.innerHTML = data.tokens.map(token => tokenCard(token, threshold)).join("");
      groupResults.forEach(group => {{
        group.cards.innerHTML = group.tokens.map(token => tokenCard(token, threshold)).join("");
      }});
      document.querySelectorAll(".badge").forEach(node => node.addEventListener("click", () => {{
        const component = Number(node.dataset.component);
        const tokenIndex = Number(node.dataset.tokenIndex);
        const same = state.selected === component && state.selectedToken === tokenIndex;
        state.selected = same ? null : component;
        state.selectedToken = same ? null : tokenIndex;
        render();
      }}));
      updateSelection();
      requestAnimationFrame(resizeFrame);
    }}
    document.getElementById("cardWidth").addEventListener("input", event => {{
      const value = Math.max(60, Math.min(360, Number(event.target.value || 100)));
      document.documentElement.style.setProperty("--card-width", `${{value}}px`);
      requestAnimationFrame(resizeFrame);
    }});
    metricSelect.addEventListener("change", event => {{
      state.metric = event.target.value;
      document.getElementById("metric").innerHTML = `<span class="chip-label">Metric:</span> ${{esc(state.metric)}}`;
      render();
    }});
    topK.addEventListener("input", event => {{
      state.topK = Math.max(1, Math.min(Number(data.component_count), Number(event.target.value || 3)));
      render();
    }});
    cutoff.addEventListener("input", render);
    document.getElementById("clear").addEventListener("click", () => {{
      state.selected = null;
      state.selectedToken = null;
      document.getElementById("selection").textContent = "Click a component to highlight it.";
      render();
    }});
    render();
    if (data.compact && isNestedFrame() && window.ResizeObserver) new ResizeObserver(resizeFrame).observe(document.body);
  </script>
</body>
</html>
"""

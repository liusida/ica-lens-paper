#!/usr/bin/env python3
"""Render a report's .html from its twin .json (same stem, e.g. top_examples.json -> .html).

Pure presentation: no model, tokenizer or GPU. Edit this file to change the UI and
rerun it; it takes about a second. The .json is written by ``neuron_activations.py``
and must contain ``top_example_windows``.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPORT_STEM = "top_examples"
DEFAULT_DATA = HERE / "L1-C79" / "neuron-activations" / f"{REPORT_STEM}.json"

MAX_ALPHA = 0.45  # strongest highlight opacity; capped so dark text stays readable

STYLE = """
:root { --bg:#ffffff; --fg:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --pos:59,130,246; --neg:249,115,22; --target:#0f172a; }
body { background:var(--bg); color:var(--fg); margin:0 auto; max-width:1200px;
  padding:24px 16px; font:14px/1.5 system-ui, sans-serif; }
h1 { font-size:20px; margin:0 0 4px; }
h2 { font-size:16px; margin:28px 0 2px; }
.meta, .lead { color:var(--muted); margin:0 0 10px; }
table { border-collapse:collapse; width:100%; }
th, td { border-bottom:1px solid var(--line); padding:5px 8px;
  vertical-align:top; text-align:left; }
th { color:var(--muted); font-weight:600; font-size:12px; }
td.num { font-variant-numeric:tabular-nums; white-space:nowrap; text-align:right; }
td.ctx { font-family:ui-monospace, Menlo, monospace; font-size:12.5px; line-height:1.9;
  white-space:pre-wrap; word-break:break-word; }
.tok { background:rgba(var(--hue), var(--a)); border-radius:3px; padding:1px 0; }
.target { outline:1px solid var(--target); outline-offset:0; font-weight:700; }
.nl { color:var(--muted); }
.legend span { display:inline-block; padding:0 6px; border-radius:3px; margin-right:4px; }
"""


def token_span(token: dict[str, Any], *, scale: float, component: int) -> str:
    value = token["activation"]
    alpha = MAX_ALPHA * min(abs(value) / scale, 1.0)
    hue = "var(--pos)" if value >= 0 else "var(--neg)"
    text = html.escape(token["text"]).replace("\n", "<span class=nl>↵</span>")
    title = f"act {value:+.3f} | C{component} {token['component_score']:+.2f}"
    classes = "tok target" if token["target"] else "tok"
    return (
        f'<span class="{classes}" title="{html.escape(title)}" '
        f'style="--a:{alpha:.3f};--hue:{hue}">{text}</span>'
    )


def neuron_section(neuron: str, entry: dict[str, Any], component: int) -> str:
    examples = entry["top_examples"]
    windows = entry["top_example_windows"]
    scale = max(abs(example["activation"]) for example in examples) or 1.0
    stats = entry["statistics"]
    rows = []
    for rank, (example, window) in enumerate(zip(examples, windows, strict=True), 1):
        spans = "".join(token_span(token, scale=scale, component=component) for token in window)
        rows.append(
            f"<tr><td class=num>{rank}</td>"
            f"<td class=num>{example['activation']:.2f}</td>"
            f"<td class=num>{example['component_score']:.2f}</td>"
            f"<td class=ctx>{spans}</td></tr>"
        )
    return (
        f"<section><h2>Neuron {neuron}</h2>"
        f"<p class=meta>writing cos {entry['writing_cos']:+.3f} · "
        f"r with C{component} score {entry['pearson_with_component_score']:+.3f} · "
        f"excess kurtosis {stats['excess_kurtosis']:.1f} · "
        f"active (&gt;0) on {100 * stats['fraction_positive']:.1f}% of tokens · "
        f"color scale: |act| / {scale:.2f}</p>"
        "<table><thead><tr><th>#</th><th>act</th>"
        f"<th>C{component}</th><th>context (every token colored by this neuron)</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
    )


def render(summary: dict[str, Any]) -> str:
    layer, component = int(summary["layer"]), int(summary["component"])
    missing = [n for n, e in summary["neurons"].items() if "top_example_windows" not in e]
    if missing:
        raise ValueError(
            f"the report .json lacks top_example_windows for neurons {missing}; "
            "run neuron_activations.py (or its --windows-only mode) first"
        )
    sections = "".join(
        neuron_section(neuron, entry, component) for neuron, entry in summary["neurons"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>L{layer} C{component} neurons</title>
<style>{STYLE}</style></head><body>
<h1>GPT-2 L{layer} MLP neurons nearest to mlp_out C{component}</h1>
<p class="lead">Top-activating tokens over {summary["tokens"]:,} profiling tokens. Every
context token is colored by the neuron's post-GELU activation at that token (blue positive,
orange negative, scaled by that neuron's top activation); the outlined token is the
example. Hover a token for its activation and C{component} score.</p>
<p class="legend"><span style="background:rgba(var(--pos),{MAX_ALPHA})">high +</span>
<span style="background:rgba(var(--pos),.15)">low +</span>
<span style="background:rgba(var(--neg),.3)">negative</span></p>
{sections}
</body></html>
"""


def render_bands(data: dict[str, Any]) -> str:
    """Random examples per activation band for one neuron (band_examples.py)."""
    layer, component, neuron = int(data["layer"]), int(data["component"]), int(data["neuron"])
    scale = float(data["color_scale"]) or 1.0
    sections = []
    for band in data["bands"]:
        rows = []
        for rank, example in enumerate(band["examples"], 1):
            spans = "".join(
                token_span(token, scale=scale, component=component) for token in example["window"]
            )
            rows.append(
                f"<tr><td class=num>{rank}</td>"
                f"<td class=num>{example['activation']:.2f}</td>"
                f"<td class=num>{example['component_score']:.2f}</td>"
                f"<td class=ctx>{spans}</td></tr>"
            )
        shown = len(band["examples"])
        sections.append(
            f"<section><h2>Activation [{band['lo']:g}, {band['hi']:g})</h2>"
            f"<p class=meta>{band['count_in_band']:,} tokens in this band; "
            f"{shown} sampled at random (sorted by activation)</p>"
            "<table><thead><tr><th>#</th><th>act</th>"
            f"<th>C{component}</th><th>context (every token colored by neuron {neuron})</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
        )
    sampling = data["sampling"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>L{layer} N{neuron} bands</title>
<style>{STYLE}</style></head><body>
<h1>GPT-2 L{layer} MLP neuron {neuron}: random examples by activation band</h1>
<p class="lead">{sampling["per_band"]} tokens drawn at random (seed {sampling["seed"]}) from
each activation band over {data["tokens"]:,} profiling tokens. Every context token is colored
by neuron {neuron}'s post-GELU activation (blue positive, orange negative; one scale for all
bands, |act| / {scale:.2f}); the outlined token is the sampled one. Hover a token for its
activation and C{component} score.</p>
<p class="legend"><span style="background:rgba(var(--pos),{MAX_ALPHA})">high +</span>
<span style="background:rgba(var(--pos),.15)">low +</span>
<span style="background:rgba(var(--neg),.3)">negative</span></p>
{"".join(sections)}
</body></html>
"""


def _kl_chart(variants: list[dict[str, Any]]) -> str:
    """Inline SVG: mean KL (log scale) against q for threshold / random / inverse."""
    series = {"threshold": "#2563eb", "random": "#94a3b8", "inverse": "#ea580c"}
    points = {
        kind: [(v["q"], v["mean_kl"]) for v in variants if v["kind"] == kind and v["mean_kl"] > 0]
        for kind in series
    }
    values = [kl for pts in points.values() for _, kl in pts]
    if not values:
        return ""
    qs = sorted({v["q"] for v in variants if v["q"] is not None})
    lo, hi = math.log10(min(values)), math.log10(max(values))
    lo, hi = math.floor(lo), math.ceil(hi) if hi > lo else math.floor(lo) + 1
    width, height, left, right, top, bottom = 640, 300, 60, 120, 16, 40

    def x(q: float) -> float:
        return left + qs.index(q) / max(len(qs) - 1, 1) * (width - left - right)

    def y(kl: float) -> float:
        return top + (hi - math.log10(kl)) / (hi - lo) * (height - top - bottom)

    parts = [f'<svg viewBox="0 0 {width} {height}" class=chart role=img>']
    for exponent in range(lo, hi + 1):
        yy = y(10.0**exponent)
        parts.append(
            f"<line x1={left} x2={width - right} y1={yy:.1f} y2={yy:.1f} class=grid />"
            f"<text x={left - 6} y={yy + 4:.1f} text-anchor=end>1e{exponent}</text>"
        )
    for q in qs:
        parts.append(f"<text x={x(q):.1f} y={height - bottom + 16} text-anchor=middle>{q:g}</text>")
    parts.append(
        f"<text x={(left + width - right) / 2} y={height - 6} text-anchor=middle>"
        "calibration percentile q</text>"
    )
    for row, (kind, color) in enumerate(series.items()):
        pts = points[kind]
        if pts:
            path = " ".join(f"{x(q):.1f},{y(kl):.1f}" for q, kl in pts)
            parts.append(f'<polyline points="{path}" fill=none stroke="{color}" stroke-width=2 />')
            parts += [
                f'<circle cx={x(q):.1f} cy={y(kl):.1f} r=3 fill="{color}" />' for q, kl in pts
            ]
        parts.append(
            f"<rect x={width - right + 14} y={top + 4 + row * 18} width=12 height=3 "
            f'fill="{color}" /><text x={width - right + 30} y={top + 9 + row * 18}>{kind}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_ablation(data: dict[str, Any]) -> str:
    """Single-neuron threshold-ablation sweep (ablation.py)."""
    layer, neuron = int(data["layer"]), int(data["neuron"])
    variants = data["variants"]
    bands = data["bands"]

    def fmt(value: float | None, spec: str) -> str:
        return "–" if value is None else format(value, spec)

    def row(v: dict[str, Any]) -> str:
        band_cells = "".join(f"<td class=num>{fmt(kl, '.1e')}</td>" for kl in v["kl_by_clean_band"])
        return (
            f"<tr class={v['kind']}><td>{html.escape(v['name'])}</td>"
            f"<td class=num>{fmt(v['threshold'], '+.3f')}</td>"
            f"<td class=num>{fmt(v['replacement'], '+.3f')}</td>"
            f"<td class=num>{100 * v['fraction_replaced']:.2f}%</td>"
            f"<td class=num><b>{v['mean_kl']:.2e}</b></td>"
            f"<td class=num>{v['mean_delta_loss']:+.2e}</td>"
            f"<td class=num>{100 * v['top1_flip_rate']:.3f}%</td>{band_cells}</tr>"
        )

    band_heads = (
        "".join(
            f"<th>KL | act in [{lo:g}, {hi:g})<br><span class=sub>n={n:,}</span></th>"
            for (lo, hi), n in zip(bands, data["band_counts"], strict=True)
        )
        .replace("[-1, ", "[&lt;0: -0.17, ")
        .replace(", 100)", ", max)")
    )
    ordered = [v for v in variants if v["kind"] == "constant"] + sorted(
        (v for v in variants if v["kind"] != "constant"),
        key=lambda v: (["threshold", "random", "inverse"].index(v["kind"]), v["q"]),
    )
    strong = data["mean_ablation_strong_positions"]
    decreased = ", ".join(
        f"{html.escape(repr(t['token']))} {t['delta_logprob']:+.2f}"
        for t in strong["most_decreased"][:15]
    )
    increased = ", ".join(
        f"{html.escape(repr(t['token']))} {t['delta_logprob']:+.2f}"
        for t in strong["most_increased"][:15]
    )
    calibration, evaluation = data["calibration"], data["evaluation"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>L{layer} N{neuron} ablation</title>
<style>{STYLE}
.chart {{ width:100%; max-width:640px; font:11px system-ui, sans-serif; fill:var(--muted); }}
.chart .grid {{ stroke:var(--line); }}
.wide {{ overflow-x:auto; }}
.sub {{ font-weight:400; }}
tr.constant td:first-child {{ font-weight:700; }}
tr.random td {{ color:var(--muted); }}
</style></head><body>
<h1>GPT-2 L{layer} MLP neuron {neuron}: threshold ablation</h1>
<p class="lead">Only neuron {neuron} is edited (before c_proj, every token). Thresholds and
replacement means come from {calibration["tokens"]:,} calibration tokens (Pile-10k profiling
sample; overall mean {calibration["mean"]:+.4f}). Effects are measured on
{evaluation["positions"]:,} held-out positions ({evaluation["dataset"]}, {evaluation["documents"]}
documents, {evaluation["precision"]}): KL(P‖P′) in nats, change in next-token loss, and top-1
flip rate. <b>threshold q</b> replaces activations below t(q) with E[a | a &lt; t];
<b>inverse q</b> replaces those at or above t(q) with E[a | a ≥ t]; <b>random q</b> replaces
the same number of tokens per document, chosen at random, with E[a | a &lt; t].</p>
<h2>Mean KL against q</h2>
{_kl_chart(variants)}
<h2>All variants</h2>
<div class=wide><table><thead><tr><th>variant</th><th>t(q)</th><th>replacement</th>
<th>replaced</th><th>mean KL</th><th>Δ loss</th><th>top-1 flips</th>{band_heads}</tr></thead>
<tbody>{"".join(row(v) for v in ordered)}</tbody></table></div>
<h2>What full mean ablation removes</h2>
<p class=meta>Average change in next-token log-probability at the {strong["positions"]:,}
positions where the neuron's clean activation is ≥ 2.</p>
<p><b>Most decreased:</b> {decreased}</p>
<p><b>Most increased:</b> {increased}</p>
</body></html>
"""


PIECE_LABELS = {
    "direct.mlp0_neurons": "layer-0 MLP neurons (direct)",
    "value.mlp0_neurons": "layer-0 MLP neurons (via L1 attention)",
    "direct.mlp0_bias": "layer-0 MLP bias (direct)",
    "value.mlp0_bias": "layer-0 MLP bias (via L1 attention)",
    "direct.emb": "token + position embedding (direct)",
    "value.emb": "token + position embedding (via L1 attention)",
    "direct.attn0": "layer-0 attention (direct)",
    "value.attn0": "layer-0 attention (via L1 attention)",
    "value.biases": "L1 attention biases and LN1 beta",
    "constant": "LN2 beta . w_in + b_in",
}


def _bar(value: float, scale: float) -> str:
    width = min(abs(value) / scale, 1.0) * 100 if scale else 0
    color = "var(--pos)" if value >= 0 else "var(--neg)"
    side = "left:50%" if value >= 0 else "right:50%"
    return (
        f'<div class=bar><span style="{side};width:{width / 2:.1f}%;'
        f'background:rgba({color},.55)"></span></div>'
    )


def _scatter(points: list[dict[str, float]]) -> str:
    if not points:
        return ""
    xs = [p["predicted"] for p in points]
    ys = [p["actual"] for p in points]
    lo, hi = min(xs + ys + [0.0]), max(xs + ys + [0.0])
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    size, margin = 320, 40

    def m(v: float) -> float:
        return margin + (v - lo) / (hi - lo) * (size - 2 * margin)

    parts = [f'<svg viewBox="0 0 {size} {size}" class=chart role=img>']
    parts.append(
        f"<line x1={m(lo):.1f} y1={size - m(lo):.1f} x2={m(hi):.1f} y2={size - m(hi):.1f} "
        'class=grid stroke-dasharray="4 3" />'
    )
    parts += [
        f'<circle cx={m(x):.1f} cy={size - m(y):.1f} r=2.5 fill="rgba(37,99,235,.6)" />'
        for x, y in zip(xs, ys, strict=True)
    ]
    parts.append(
        f"<text x={size / 2} y={size - 8} text-anchor=middle>predicted Δz (frozen linear)</text>"
        f'<text x=12 y={size / 2} text-anchor=middle transform="rotate(-90 12 {size / 2})">'
        "actual Δz (model)</text>"
    )
    for v in (lo, 0.0, hi):
        parts.append(f"<text x={m(v):.1f} y={size - margin + 14} text-anchor=middle>{v:.1f}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _attribution_span(token: dict[str, Any], scale: float) -> str:
    value = token["activation"]
    alpha = MAX_ALPHA * min(abs(value) / scale, 1.0)
    hue = "var(--pos)" if value >= 0 else "var(--neg)"
    text = html.escape(token["text"]).replace("\n", "<span class=nl>↵</span>")
    classes = "tok target" if token["target"] else "tok"
    return (
        f'<span class="{classes}" title="act {value:+.3f}" '
        f'style="--a:{alpha:.3f};--hue:{hue}">{text}</span>'
    )


def render_input_attribution(data: dict[str, Any]) -> str:
    """Exact frozen-LN/attention attribution of a L1 neuron to L0 pieces (input_attribution.py)."""
    layer, neuron = int(data["layer"]), int(data["neuron"])
    selection = data["selection"]
    pieces = data["mean_pieces"]
    scale = max(abs(v) for v in pieces.values()) or 1.0
    piece_rows = "".join(
        f"<tr><td>{PIECE_LABELS.get(key, key)}</td><td class=num>{value:+.3f}</td>"
        f"<td class=barcell>{_bar(value, scale)}</td></tr>"
        for key, value in sorted(pieces.items(), key=lambda item: -abs(item[1]))
    )
    neuron_rows = "".join(
        f"<tr><td class=num>{t['neuron']}</td><td class=num>{t['mean_direct']:+.3f}</td>"
        f"<td class=num>{t['mean_value']:+.3f}</td><td class=num><b>{t['mean_total']:+.3f}</b></td>"
        f"<td class=num>{100 * t['top5_fraction']:.1f}%</td>"
        f"<td class=ctx>{html.escape(' · '.join(repr(x) for x in t['top5_tokens']))}</td></tr>"
        for t in data["top_neurons"]
    )
    validation = data["validation"]
    pearson = "–" if validation["pearson"] is None else format(validation["pearson"], ".3f")
    concentration = data["concentration"]
    cards = []
    for example in data["examples"]:
        window = "".join(_attribution_span(token, 4.0) for token in example["window"])
        top_rows = "".join(
            f"<tr><td class=num>{t['neuron']}</td><td class=num>{t['a0']:.2f}</td>"
            f"<td class=num>{t['direct']:+.3f}</td><td class=num>{t['value']:+.3f}</td>"
            f"<td class=num><b>{t['direct'] + t['value']:+.3f}</b></td></tr>"
            for t in example["top_neurons"]
        )
        p = example["pieces"]
        biases = p["direct.mlp0_bias"] + p["value.mlp0_bias"] + p["value.biases"] + p["constant"]
        summary = (
            f"neurons direct {p['direct.mlp0_neurons']:+.2f} · neurons via attn "
            f"{p['value.mlp0_neurons']:+.2f} · embedding {p['direct.emb'] + p['value.emb']:+.2f}"
            f" · L0 attention {p['direct.attn0'] + p['value.attn0']:+.2f} · biases/constants "
            f"{biases:+.2f}"
        )
        sources = " · ".join(
            f"{html.escape(repr(s['token']))} ({s['offset']:+d}) {s['value']:+.2f}"
            for s in example["value_sources"]
        )
        cards.append(
            f"<div class=card><div class=ctx>{window}</div>"
            f"<p class=meta><b>{html.escape(repr(example['token']))}</b> · z {example['z']:.2f}"
            f" · act {example['activation']:.2f} — {summary}</p>"
            f"<p class=meta>via L1 attention, strongest source tokens: {sources or '–'}</p>"
            "<table class=small><thead><tr><th>L0 neuron</th><th>a0</th><th>direct</th>"
            f"<th>via attn</th><th>total</th></tr></thead><tbody>{top_rows}</tbody></table></div>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>L{layer} N{neuron} input attribution</title>
<style>{STYLE}
.chart {{ width:100%; max-width:340px; font:11px system-ui, sans-serif; fill:var(--muted); }}
.chart .grid {{ stroke:var(--muted); }}
.bar {{ position:relative; height:12px; background:linear-gradient(var(--line),var(--line))
  center/1px 100% no-repeat; min-width:200px; }}
.bar span {{ position:absolute; top:0; bottom:0; border-radius:2px; }}
td.barcell {{ width:40%; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:12px 0; }}
.card .ctx {{ font-family:ui-monospace, Menlo, monospace; font-size:12.5px; line-height:1.9;
  white-space:pre-wrap; word-break:break-word; }}
table.small {{ width:auto; }}
table.small td, table.small th {{ padding:2px 10px; font-size:12px; }}
.grid2 {{ display:flex; gap:24px; flex-wrap:wrap; align-items:flex-start; }}
code {{ font-size:12.5px; }}
</style></head><body>
<h1>GPT-2 L{layer} neuron {neuron}: what feeds it</h1>
<p class="lead">Pre-activation z of neuron {neuron} at the {selection["tokens"]:,} profiling
tokens with activation ≥ t(q{selection["quantile"]:g}) = {selection["threshold"]:.3f},
decomposed into layer-0 pieces. With both LayerNorm scales and the layer-1 attention pattern
frozen at their actual values the decomposition is exact (max |sum − z| =
{data["max_reconstruction_error"]:.1e}). A layer-0 neuron j reaches z directly,
<code>a_j(t)·(u·W0_out[j])/σ2</code>, or through layer-1 attention values from earlier tokens,
<code>Σ_h Σ_s A_h[t,s] a_j(s)(k_h·W0_out[j])/(σ1_s σ2)</code>. Mean z = {data["mean_z"]:.3f}.</p>
<h2>Where z comes from (mean over tokens)</h2>
<table><tbody>{piece_rows}</tbody></table>
<h2>How concentrated is the layer-0 neuron input?</h2>
<p class=meta>Per token, sorting layer-0 neurons by |contribution|: the median token needs
<b>{concentration["k50_median"]:.0f}</b> neurons for 50% and
<b>{concentration["k90_median"]:.0f}</b> for 90% of the total |contribution|
(90th percentile of tokens: {concentration["k50_p90"]:.0f} and {concentration["k90_p90"]:.0f}).
Sum of mean neuron contributions: direct {data["neuron_totals"]["mean_direct_sum"]:+.3f},
via attention {data["neuron_totals"]["mean_value_sum"]:+.3f}.</p>
<h2>Top layer-0 neurons (by mean contribution)</h2>
<table><thead><tr><th>L0 neuron</th><th>mean direct</th><th>mean via attn</th><th>mean total</th>
<th>in token's top-5</th><th>target tokens where it is top-5</th></tr></thead>
<tbody>{neuron_rows}</tbody></table>
<h2>Does the linear attribution predict ablations?</h2>
<div class=grid2>{_scatter(validation["points"])}
<p class=meta>{validation["tokens"]} random tokens; {html.escape(validation["ablation"])}.
Pearson r = {pearson};
mean predicted Δz {validation["mean_predicted"]:+.3f}, mean actual Δz
{validation["mean_actual"]:+.3f}. The gap reflects the frozen σ and attention pattern.</p></div>
<h2>Examples ({len(data["examples"])} tokens, random per activation band)</h2>
<p class=meta>Context colored by neuron {neuron}'s activation; the outlined token is the
example.</p>
{"".join(cards)}
</body></html>
"""


def _pair_line(window: list[dict[str, Any]], key: str, scale: float, label: str) -> str:
    spans = []
    for token in window:
        value = token[key]
        alpha = MAX_ALPHA * min(abs(value) / scale, 1.0)
        hue = "var(--pos)" if value >= 0 else "var(--neg)"
        text = html.escape(token["text"]).replace("\n", "<span class=nl>↵</span>")
        classes = "tok target" if token["target"] else "tok"
        spans.append(
            f'<span class="{classes}" title="{label} {value:+.3f}" '
            f'style="--a:{alpha:.3f};--hue:{hue}">{text}</span>'
        )
    body = "".join(spans)
    return f"<div class=line><span class=who>{label}</span><span class=ctx>{body}</span></div>"


def render_pair_examples(data: dict[str, Any]) -> str:
    """Both / only-one / neither examples for two neurons (pair_examples.py)."""
    up, down = data["up"], data["down"]
    up_label = f"L{up['layer']} N{up['neuron']}"
    down_label = f"L{down['layer']} N{down['neuron']}"
    scale = data["color_scale"]
    sections = []
    for group in data["groups"]:
        cards = "".join(
            "<div class=card>"
            f"<p class=meta><b>{html.escape(repr(e['token']))}</b> · {up_label} {e['up']:.2f}"
            f" · {down_label} {e['down']:.2f}</p>"
            + _pair_line(e["window"], "up", scale["up"], up_label)
            + _pair_line(e["window"], "down", scale["down"], down_label)
            + "</div>"
            for e in group["examples"]
        )
        sections.append(
            f"<section><h2>{html.escape(group['name'])}</h2>"
            f"<p class=meta>{group['tokens_in_group']:,} tokens in this group; "
            f"{len(group['examples'])} sampled at random</p>{cards}</section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{up_label} vs {down_label}</title>
<style>{STYLE}
.card {{ border:1px solid var(--line); border-radius:8px; padding:8px 12px; margin:10px 0; }}
.card .meta {{ margin:0 0 4px; }}
.line {{ display:flex; gap:10px; align-items:baseline; }}
.who {{ flex:0 0 72px; color:var(--muted); font-size:12px; }}
.line .ctx {{ font-family:ui-monospace, Menlo, monospace; font-size:12.5px; line-height:1.9;
  white-space:pre-wrap; word-break:break-word; }}
</style></head><body>
<h1>{up_label} and {down_label}: when do they fire together?</h1>
<p class="lead">"Fires" means activation ≥ that neuron's q{data["quantile"]:g} over
{data["tokens"]:,} profiling tokens: {up_label} ≥ {up["threshold"]:.3f}, {down_label} ≥
{down["threshold"]:.3f}. Each example shows its context twice, colored by each neuron's
post-GELU activation (each on its own scale: |act| / {scale["up"]:.2f} and
/ {scale["down"]:.2f}); the outlined token is the sampled one.</p>
<p class="legend"><span style="background:rgba(var(--pos),{MAX_ALPHA})">high +</span>
<span style="background:rgba(var(--pos),.15)">low +</span>
<span style="background:rgba(var(--neg),.3)">negative</span></p>
{"".join(sections)}
</body></html>
"""


def render_label_test(data: dict[str, Any]) -> str:
    """Hand-written hypothesis tests for two neuron labels (label_test.py)."""
    up, down = data["up"], data["down"]
    up_label = f"L{up['layer']} N{up['neuron']}"
    down_label = f"L{down['layer']} N{down['neuron']}"
    scale = data["color_scale"]
    colors = {"pass": "#15803d", "partial": "#b45309", "fail": "#b91c1c", "open": "#64748b"}

    def badge(case: dict[str, Any], key: str) -> str:
        v = case["verdict"][key]
        return (
            f"<span class=badge style='color:{colors[v['status']]}'>"
            f"pred {case['prediction'][key]} · fired {v['fired']}/{v['n']} · {v['status']}</span>"
        )

    rows = "".join(
        f"<tr><td>{html.escape(c['title'])}</td><td>{html.escape(c['tests'])}</td>"
        f"<td>{badge(c, 'up')}</td><td>{badge(c, 'down')}</td></tr>"
        for c in data["cases"]
    )
    sections = []
    for case in data["cases"]:
        table = "".join(
            f"<tr><td>{html.escape(e['word'])}</td>"
            f"<td class=mono>{html.escape('|'.join(e['pieces']))}</td>"
            f"<td class=num>{e['up']:+.2f}</td><td class=num>{e['down']:+.2f}</td>"
            f"<td>{_pair_line(e['window'], 'up', scale['up'], up_label)}"
            f"{_pair_line(e['window'], 'down', scale['down'], down_label)}</td></tr>"
            for e in case["examples"]
        )
        sections.append(
            f"<section><h2>{html.escape(case['title'])}</h2>"
            f"<p class=meta>{html.escape(case['tests'])} — {up_label}: {badge(case, 'up')}"
            f" · {down_label}: {badge(case, 'down')}</p>"
            f"<table><tr><th>word</th><th>tokens</th><th>{up_label}</th><th>{down_label}</th>"
            f"<th>context</th></tr>{table}</table></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label test {up_label} {down_label}</title>
<style>{STYLE}
.badge {{ font-size:12px; font-weight:600; white-space:nowrap; }}
.mono {{ font-family:ui-monospace, Menlo, monospace; font-size:12.5px; }}
.line {{ display:flex; gap:10px; align-items:baseline; }}
.who {{ flex:0 0 72px; color:var(--muted); font-size:12px; }}
.line .ctx {{ font-family:ui-monospace, Menlo, monospace; font-size:12.5px; line-height:1.9;
  white-space:pre-wrap; word-break:break-word; }}
</style></head><body>
<h1>Testing labels for {up_label} and {down_label}</h1>
<p class="lead">{up_label}: <b>{html.escape(up["label"])}</b>. {down_label}:
<b>{html.escape(down["label"])}</b>. Each case is a set of hand-written sentences, and both
neurons are read at the last token of the target word. "Fired" means the activation is ≥ that
neuron's q99 over 1M profiling tokens ({up_label} ≥ {up["threshold"]:.3f}, {down_label} ≥
{down["threshold"]:.3f}). A prediction of high passes only if every sentence fires, low only if
none does, and "?" is an open question that is not scored.</p>
<table><tr><th>case</th><th>tests</th><th>{up_label}</th><th>{down_label}</th></tr>{rows}</table>
{"".join(sections)}
</body></html>
"""


COMPARISON_METRICS = (
    ("excess_kurtosis", "excess kurtosis", "{:.1f}", "higher = heavier tail"),
    ("fraction_positive", "fraction active (&gt;0)", "{:.3f}", "lower = sparser"),
    (
        "best_abs_r",
        "best |r| with one ICA component",
        "{:.3f}",
        "higher = one component explains it",
    ),
    ("second_abs_r", "2nd-best |r| with a component", "{:.3f}", "lower = less mixed"),
    ("best_abs_cos", "best |cos| writing dir vs ICA dir", "{:.3f}", "higher = aligned"),
    ("effective_types", "effective token types in top 1%", "{:.0f}", "lower = more token-specific"),
    (
        "top10_type_share",
        "top-10 token types' share of top 1%",
        "{:.3f}",
        "higher = more token-specific",
    ),
)


def _strip(values: list[float], ref: float, rnd: float, log: bool) -> str:
    import math

    f = (lambda v: math.log10(max(v, 1e-9))) if log else (lambda v: v)
    xs = [f(v) for v in values + [ref, rnd]]
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0

    def x(v: float) -> float:
        return 6 + 288 * (f(v) - lo) / span

    dots = "".join(
        f'<circle cx="{x(v):.1f}" cy="{12 + (i % 5) * 2}" r="2" fill="#94a3b8" opacity=".6"/>'
        for i, v in enumerate(values)
    )
    return (
        f'<svg width="300" height="30" role="img">{dots}'
        f'<line x1="{x(rnd):.1f}" x2="{x(rnd):.1f}" y1="2" y2="28" '
        f' stroke="#f97316" stroke-width="2"/>'
        f'<line x1="{x(ref):.1f}" x2="{x(ref):.1f}" y1="2" y2="28" '
        f' stroke="#2563eb" stroke-width="2"/>'
        "</svg>"
    )


def _example_rows(examples: list[dict[str, Any]], scale: float, component: int) -> str:
    rows = []
    for rank, example in enumerate(examples, 1):
        spans = "".join(token_span(t, scale=scale, component=component) for t in example["window"])
        rows.append(
            f"<tr><td class=num>{rank}</td><td class=num>{example['activation']:.2f}</td>"
            f"<td class=num>{example['component_score']:.2f}</td><td class=ctx>{spans}</td></tr>"
        )
    return "".join(rows)


def render_random_comparison(data: dict[str, Any]) -> str:
    """Reference neuron vs a random neuron vs a random population (random_neuron.py)."""
    layer, ref_n, rnd_n = int(data["layer"]), int(data["reference"]), int(data["random_neuron"])
    metrics = data["metrics"]
    ref, rnd, pop = metrics[0], metrics[data.get("random_index", 1)], metrics[1:]
    rows = []
    for key, label, spec, note in COMPARISON_METRICS:
        values = [m[key] for m in pop]
        pct = 100 * sum(v < ref[key] for v in values) / len(values)
        pct_r = 100 * sum(v < rnd[key] for v in values) / len(values)
        median = sorted(values)[len(values) // 2]
        log = key in ("excess_kurtosis", "effective_types")
        rows.append(
            f"<tr><td>{label}<div class=meta>{note}</div></td>"
            f"<td class=num><b>{spec.format(ref[key])}</b><div class=meta>pct {pct:.0f}</div></td>"
            f"<td class=num>{spec.format(rnd[key])}<div class=meta>pct {pct_r:.0f}</div></td>"
            f"<td class=num>{spec.format(median)}</td>"
            f"<td>{_strip(values, ref[key], rnd[key], log)}</td></tr>"
        )
    comp_rows = "".join(
        f"<tr><td>N{m['neuron']}</td><td class=num>C{m['best_component_by_r']} "
        f"(r {m['best_r_sign'] * m['best_abs_r']:+.3f})</td>"
        f"<td class=num>C{m['best_component_by_cos']} (|cos| {m['best_abs_cos']:.3f})</td>"
        f"<td class=num>{m['q99_threshold']:.3f}</td><td class=num>{m['max']:.2f}</td></tr>"
        for m in (ref, rnd)
    )
    sections = []
    for entry in data["examples"]:
        n, comp, scale = entry["neuron"], entry["score_component"], entry["color_scale"] or 1.0
        head = (
            f"<table><thead><tr><th>#</th><th>act</th><th>C{comp}</th>"
            f"<th>context (every token colored by N{n})</th></tr></thead><tbody>"
        )
        band_html = "".join(
            f"<h3>Percentile [{b['q_lo']:g}, {b['q_hi']:g}) · act [{b['lo']:.2f}, {b['hi']:.2f})"
            f" · {b['count']:,} tokens</h3>{head}"
            f"{_example_rows(b['examples'], scale, comp)}</tbody></table>"
            for b in entry["bands"]
        )
        sections.append(
            f"<section><h2>Neuron {n}" + (" (reference)" if n == ref_n else " (random)") + "</h2>"
            f"<p class=meta>C{comp} column = score of this neuron's best-correlated ICA component "
            f"· color scale |act| / {scale:.2f}</p>"
            f"<h3>Top {len(entry['top_examples'])}</h3>{head}"
            f"{_example_rows(entry['top_examples'], scale, comp)}</tbody></table>"
            f"{band_html}</section>"
        )
    population = data["population"]
    dead = 100 * population.get("dead_fraction", 0)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>L{layer} N{ref_n} vs random</title>
<style>{STYLE} h3 {{ font-size:13px; margin:16px 0 2px; color:var(--muted); }}</style></head><body>
<h1>Is GPT-2 L{layer} N{ref_n} cleaner than a random L{layer} neuron?</h1>
<p class="lead">N{ref_n} (blue) was selected as the neuron nearest to the clean ICA component
C79. N{rnd_n} (orange) is a random L{layer} neuron; grey dots are {population["size"]} random
L{layer} neurons (seed {population["seed"]}, excluding C79's five nearest;
{dead:.0f}% never exceed 0.05), all measured on the
same {data["tokens"]:,} profiling tokens. "pct" = percentile within the random population.</p>
<table><thead><tr><th>metric</th><th>N{ref_n}</th><th>N{rnd_n}</th><th>pop median</th>
<th>population (grey) · <span style="color:#2563eb">N{ref_n}</span> ·
<span style="color:#f97316">N{rnd_n}</span></th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Best-matching ICA components</h2>
<table><thead><tr><th>neuron</th><th>best by activation r</th><th>best by writing cos</th>
<th>q99 activation</th><th>max activation</th></tr></thead><tbody>{comp_rows}</tbody></table>
<p class="legend"><span style="background:rgba(var(--pos),{MAX_ALPHA})">high +</span>
<span style="background:rgba(var(--pos),.15)">low +</span>
<span style="background:rgba(var(--neg),.3)">negative</span></p>
{"".join(sections)}
</body></html>
"""


def render_any(data: dict[str, Any]) -> str:
    """Dispatch on the report kind stored in the .json."""
    if data.get("kind") == "random_neuron_comparison":
        return render_random_comparison(data)
    if data.get("kind") == "neuron_label_test":
        return render_label_test(data)
    if data.get("kind") == "neuron_pair_examples":
        return render_pair_examples(data)
    if data.get("kind") == "neuron_input_attribution":
        return render_input_attribution(data)
    if data.get("kind") == "neuron_ablation":
        return render_ablation(data)
    if data.get("kind") == "neuron_bands":
        return render_bands(data)
    return render(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, nargs="?", default=DEFAULT_DATA)
    args = parser.parse_args()
    summary = json.loads(args.data.read_text())
    output = args.data.with_suffix(".html")  # twin of the .json: same stem
    output.write_text(render_any(summary))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

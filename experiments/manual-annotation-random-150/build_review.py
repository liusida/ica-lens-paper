"""Build one review page containing all sampled ICA component profiles."""

# ruff: noqa: E501 -- Long lines keep the embedded HTML template readable.

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from icalens import ICALens
from icalens.html import component_profile_html

ROOT = Path("experiments/manual-annotation-random-150")
RESULTS = ROOT / "results"
LENSES = {
    "gpt2": Path("local-icalens-models/official/icalens-gpt2-small-pile10k"),
    "gemma-2-2b": Path("local-icalens-models/official/icalens-gemma-2-2b-pile10k"),
    "qwen3.5-9b": Path(
        "local-icalens-models/official/icalens-qwen3.5-9b-base-pile10k"
    ),
}
RANK_THRESHOLDS = (1, 3, 5, 10, 15)
HEADLINE_THRESHOLD = 15
OCCURRENCES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-erf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_path = RESULTS / "review.html"
    if review_path.exists() and not args.force:
        raise FileExistsError("review page already exists; pass --force to replace it")

    sections = []
    total = 0
    profile_styles: str | None = None
    for model, lens_path in LENSES.items():
        lens = ICALens.from_pretrained(lens_path)
        rows = _read_selection(RESULTS / model / "components.csv")
        cards = []
        for row in rows:
            layer = int(row["layer"])
            component = int(row["component_id"])
            erf = _load_or_measure_erf(
                lens,
                model=model,
                layer=layer,
                component=component,
                refresh=args.refresh_erf,
            )
            profile = lens.component_profile(layer=layer, component=component)
            styles, panel = _profile_fragment(profile, layer=layer)
            if profile_styles is None:
                profile_styles = styles
            elif styles != profile_styles:
                raise RuntimeError("component profile panels returned inconsistent styles")
            item_id = f"{model}-L{layer:02d}-C{component:05d}"
            cards.append(
                f"""
<article class="card" id="{html.escape(item_id)}" data-model="{html.escape(model)}">
  <header>
    <a href="#{html.escape(item_id)}">{html.escape(model)} · layer {layer} · C{component} · {_erf_label(erf)}</a>
  </header>
  {panel}
</article>"""
            )
        total += len(cards)
        sections.append(
            f"""
<section id="{html.escape(model)}">
  <h2>{html.escape(model)} <span>{len(cards)} components</span></h2>
  {''.join(cards)}
</section>"""
        )
        lens.unload_model()

    if profile_styles is None:
        raise RuntimeError("no component profile styles were rendered")
    review_path.write_text(
        _document(sections, total, profile_styles=profile_styles), encoding="utf-8"
    )
    print(f"Wrote {total} component panels to {review_path}")


def _read_selection(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50:
        raise ValueError(f"expected 50 sampled components in {path}, found {len(rows)}")
    return rows


def _load_or_measure_erf(
    lens: ICALens,
    *,
    model: str,
    layer: int,
    component: int,
    refresh: bool,
) -> dict[str, Any]:
    path = RESULTS / model / "erf" / f"layer_{layer:02d}-C{component:05d}.json"
    if path.is_file() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        _validate_erf(cached, model=model, layer=layer, component=component)
        print(f"ERF cache hit: {model} layer {layer} C{component}", flush=True)
        return cached
    print(f"Measuring ERF: {model} layer {layer} C{component}", flush=True)
    result = lens.erf.suffix_sweep(
        layer=layer,
        component=component,
        rank_thresholds=RANK_THRESHOLDS,
        occurrences=OCCURRENCES,
        verbose=True,
    )
    result["review_model"] = model
    result["requested_occurrences"] = OCCURRENCES
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def _validate_erf(
    result: dict[str, Any], *, model: str, layer: int, component: int
) -> None:
    expected = {
        "review_model": (result.get("review_model"), model),
        "layer": (result.get("layer"), layer),
        "component": (result.get("component"), component),
        "rank_thresholds": (result.get("rank_thresholds"), list(RANK_THRESHOLDS)),
        "requested_occurrences": (
            result.get("requested_occurrences"),
            OCCURRENCES,
        ),
    }
    mismatches = [
        f"{key}: {actual!r} != {wanted!r}"
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(f"incompatible ERF cache: {'; '.join(mismatches)}")


def _erf_label(result: dict[str, Any]) -> str:
    threshold = result["threshold_results"][str(HEADLINE_THRESHOLD)]
    mean = float(threshold["suffix_erf_mean"])
    return f"top-{HEADLINE_THRESHOLD} suffix ERF: {mean:.2f} tokens"


def _profile_fragment(profile: object, *, layer: int) -> tuple[str, str]:
    document = component_profile_html(profile, layer=layer)
    style_start = document.index("<style>") + len("<style>")
    style_end = document.index("</style>", style_start)
    body_start = document.index("<body>", style_end) + len("<body>")
    body_end = document.index("<script>", body_start)
    return document[style_start:style_end].strip(), document[body_start:body_end].strip()


def _document(sections: list[str], total: int, *, profile_styles: str) -> str:
    links = "".join(
        f'<a href="#{html.escape(model)}">{html.escape(model)}</a>' for model in LENSES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manual annotation · {total} random ICA components</title>
<style>
{profile_styles}
  :root{{--bg:#eef1f5;--panel:#fff;--text:#151922;--muted:#647084;--border:#cbd3df}}
  *{{box-sizing:border-box}} html{{scroll-behavior:smooth}}
  body{{margin:0;padding:0;overflow:auto;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
  nav{{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:18px;padding:10px 18px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--border)}}
  nav strong{{margin-right:auto}} nav a{{color:#315f9b;text-decoration:none;font-weight:650}}
  main{{max-width:1500px;margin:0 auto;padding:18px}}
  section{{scroll-margin-top:58px}} h2{{margin:24px 0 10px}} h2 span{{color:var(--muted);font-size:13px;font-weight:500}}
  .card{{margin:0 0 14px;border:1px solid var(--border);border-radius:9px;background:var(--panel);overflow:hidden;scroll-margin-top:58px}}
  .card header{{display:flex;justify-content:space-between;gap:12px;padding:9px 13px;border-bottom:1px solid var(--border);font-weight:700}}
  .card header a{{color:#273244;text-decoration:none}}
  .card > .panel{{margin:6px;border:0}}
</style>
</head>
<body>
<nav><strong>{total} sampled ICA component profiles</strong>{links}</nav>
<main>{''.join(sections)}</main>
</body>
</html>
"""


if __name__ == "__main__":
    main()

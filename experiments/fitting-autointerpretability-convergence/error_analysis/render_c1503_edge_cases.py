#!/usr/bin/env python3
"""Render the saved C1503 edge-case probe as a self-contained HTML report."""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results/c1503-edge-case-results.json"
OUTPUT = ROOT / "results/c1503-edge-case-results.html"


def token_cell(row: dict, *, target: bool, maximum: bool, scale: float) -> str:
    activation = float(row["activation"])
    signed = float(row["signed_score"])
    intensity = min(1.0, activation / scale) if scale > 0 else 0.0
    background = f"rgba(55, 96, 157, {0.06 + 0.34 * intensity:.3f})"
    classes = ["token"]
    if target:
        classes.append("target")
    if maximum:
        classes.append("maximum")
    decoded = str(row["decoded"])
    visible = decoded.replace(" ", "␠") or "∅"
    tooltip = (
        f"position={row['position']}; token_id={row['token_id']}; "
        f"tokenizer={row['token']!r}; signed={signed:.6f}; "
        f"activation={activation:.6f}"
    )
    return (
        f'<span class="{" ".join(classes)}" style="background:{background}" '
        f'title="{html.escape(tooltip, quote=True)}">'
        f'<span class="surface">{html.escape(visible)}</span>'
        f'<span class="score">{activation:.2f}</span>'
        f'<span class="position">{row["position"]}</span>'
        "</span>"
    )


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    records = data["records"]
    global_max = max(
        float(token["activation"])
        for record in records
        for token in record["tokens"]
    )
    cards = []
    for record in records:
        target_positions = {
            int(token["position"]) for token in record.get("focus_tokens", [])
        }
        maximum_position = int(record["maximum"]["position"])
        tokens = "".join(
            token_cell(
                token,
                target=int(token["position"]) in target_positions,
                maximum=int(token["position"]) == maximum_position,
                scale=global_max,
            )
            for token in record["tokens"]
        )
        target_note = (
            "Target tokens are underlined. "
            if target_positions
            else "The focus text is split across tokenizer tokens. "
        )
        cards.append(
            f"""
            <section class="case">
              <header>
                <span class="case-number">Case {record['case']}</span>
                <span class="group">{html.escape(record['group'])}</span>
                <span class="focus">focus: {html.escape(record['focus'])}</span>
              </header>
              <p class="prompt">{html.escape(record['prompt'])}</p>
              <div class="tokens">{tokens}</div>
              <p class="note">{target_note}The blue outline marks the maximum activation.</p>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen iteration-50 L15/C1503 edge cases</title>
<style>
  :root {{ color-scheme: light; font-family: Georgia, "Times New Roman", serif; }}
  body {{ margin: 0; color: #20242a; background: #f5f6f8; }}
  main {{ width: min(1180px, calc(100% - 40px)); margin: 32px auto 64px; }}
  h1 {{ margin-bottom: 8px; font-size: 28px; }}
  .subtitle {{ margin: 0 0 24px; color: #555d68; line-height: 1.45; }}
  .legend {{ display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
             margin-bottom: 22px; font-family: Arial, sans-serif; font-size: 13px; }}
  .legend-target {{ text-decoration: underline 3px #b6543f; text-underline-offset: 4px; }}
  .legend-max {{ padding: 3px 6px; border: 2px solid #315f9b; border-radius: 4px; }}
  .case {{ margin: 14px 0; padding: 16px 18px 14px; background: white;
           border: 1px solid #d9dde3; border-radius: 8px; box-shadow: 0 1px 2px #0000000a; }}
  header {{ display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
  .case-number {{ font-weight: 700; font-size: 17px; }}
  .group, .focus {{ font-family: Arial, sans-serif; font-size: 12px; color: #59616d;
                    background: #eef1f5; padding: 3px 7px; border-radius: 999px; }}
  .prompt {{ font-size: 17px; margin: 10px 0 14px; }}
  .tokens {{ display: flex; align-items: flex-end; flex-wrap: wrap; gap: 6px 4px;
             font-family: Arial, sans-serif; }}
  .token {{ position: relative; display: inline-grid; grid-template-rows: auto auto;
            justify-items: center; min-width: 30px; padding: 7px 5px 5px;
            border: 2px solid transparent; border-radius: 5px; }}
  .token.target .surface {{ text-decoration: underline 3px #b6543f;
                            text-underline-offset: 4px; }}
  .token.maximum {{ border-color: #315f9b; }}
  .surface {{ font-size: 15px; line-height: 1.2; white-space: pre; }}
  .score {{ margin-top: 7px; font: 700 12px/1 Arial, sans-serif; color: #22344d; }}
  .position {{ position: absolute; top: 1px; right: 3px; font: 9px/1 Arial, sans-serif;
               color: #858c96; }}
  .note {{ margin: 9px 0 0; color: #737b86; font: 11px/1.3 Arial, sans-serif; }}
</style>
</head>
<body><main>
  <h1>Qwen iteration-50 · L15/C1503</h1>
  <p class="subtitle">Each box is one tokenizer token. The number below it is the
  nonnegative oriented activation used by the autointerpretability evaluator.
  Hover for token ID, tokenizer text, and the unclipped signed score.</p>
  <div class="legend">
    <span class="legend-target">designated target</span>
    <span class="legend-max">maximum activation</span>
    <span>␠ denotes a leading space</span>
  </div>
  {''.join(cards)}
</main></body></html>
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

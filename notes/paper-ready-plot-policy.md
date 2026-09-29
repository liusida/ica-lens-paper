# Paper-ready plot policy

This document records conventions for figures embedded in the paper. It
complements [`plot-style-policy.md`](plot-style-policy.md), which remains the
general policy for experiment figures and previews.

## Understand the figure before changing it

- Before changing a figure, state what question it answers and what comparison,
  mechanism, or result the reader should notice first.
- Trace each visual element back to its meaning in the data: identify what the
  points, lines, colors, ordering, annotations, reference fits, and highlighted
  examples represent.
- Evaluate a proposed change by whether it strengthens that intended reading.
  Do not optimize an element in isolation if doing so weakens the relationship
  between panels or changes the scientific emphasis.
- Preserve established encodings and unaffected details. Make surgical changes
  unless the figure's intended message requires a broader redesign.
- When the intended message is unclear, discuss and agree on it before editing.
  A visually cleaner figure is not necessarily a more truthful or useful one.
- After editing, read the figure again from a new reader's perspective and
  verify that the intended conclusion is easier—not merely prettier—to see.

## Separate paper rendering from experiment rendering

- Keep the existing experiment plotting scripts responsible for PNGs used in
  websites, notebooks, slides, and experiment reports.
- Put paper-only rendering code under
  `paper/overleaf-v2.1/scripts/`, with one module per figure and shared visual
  constants in `scripts/style.py`.
- Apply the same separation to tables. Put paper-only table renderers under
  `paper/overleaf-v2.1/scripts/tables/`; they should consume frozen paper data
  and generate the corresponding LaTeX fragments under `tables/`.
- `scripts/prepare_data.py` may read source results directly from the project
  by relative path. It should reorganize them into the exact representation
  consumed by the paper figure or table; it must not rerun an experiment.
- Before preparing a data twin, understand and state the figure or table's
  intended meaning: what question it answers, which comparison or evidence
  supports that answer, and what each displayed quantity represents. Trace the
  source computation far enough to verify that its logic actually measures the
  intended quantity.
- Audit the source data during preparation whenever practical. Check run
  completion, expected coverage, uniqueness, missing or non-finite values,
  aggregation denominators, normalization, and relevant invariants. Also check
  semantic alignment: the source logic, frozen representation, visual encoding,
  caption, and intended claim must describe the same population and operation.
  A structurally complete dataset is not sufficient if it answers a different
  question from the figure.
- Store one compact data twin beside each paper figure, using the same stem:
  for example, `figures/overview.npz` and `figures/overview.pdf`. The data twin
  is plot-specific and should be sufficient to reproduce the corresponding
  PDF without reopening the full experiment output.
- Store one compact JSON twin beside each generated paper table, using the same
  semantic stem: for example, `tables/component-profile-examples.json` and
  `tables/component-profile-examples.tex`. Use JSON rather than NPZ for tables
  because table rows combine strings, identifiers, nested records, provenance,
  and a small number of scalar values. The JSON twin must be sufficient to
  reproduce the LaTeX fragment without reopening experiment outputs or model
  artifacts.
- Table twins should preserve the complete audited values needed for the
  presentation, such as full contexts and exact target spans. Truncation,
  abbreviation, wrapping, escaping, and emphasis belong in the LaTeX renderer,
  not in the twin.
- When a table combines several artifacts, validate their semantic join before
  freezing it: component or example identities must agree, requested entries
  must be unique, required fields must be complete, and metric definitions and
  aggregation settings must match the caption. Record source paths and content
  hashes in the JSON twin whenever practical.
- Data preparation and rendering are separate explicit operations. Require
  `--force` before replacing a frozen data twin, rendered figure, or generated
  LaTeX table fragment.
- `scripts/make_all.py` is the single entry point for regenerating paper
  figures. Debug decorations must be opt-in and absent from final output.

## Use semantic identities, not figure or table numbers

- Name renderers, preparation targets, data twins, PDFs, table fragments, and
  LaTeX labels by scientific role rather than current display number. Prefer
  `historical_ica_sae.py`, `--figure historical-ica-sae`, and
  `fig:historical-baseline` over names containing `figure_04` or `figure-4`.
- LaTeX alone owns displayed figure and table numbers. Inserting or reordering
  an item must not change the meaning of a filename or command.
- Preserve intended regeneration order in an explicit ordered registry in
  `prepare_data.py` and `make_all.py`. Stable identity and mutable paper order
  are separate concerns.
- Use the same semantic stem for the data twin and rendered artifact whenever
  practical, such as `ica-fitting-budget-comparison.npz` and
  `ica-fitting-budget-comparison.pdf`.

## Version alternative figure designs through generators

- When comparing alternative designs for the same numbered figure, version
  the generation scripts rather than the generated artifacts. For example,
  keep `overview_v1.py` and `overview.py`, while both read
  `figures/overview.npz` and write the canonical `figures/overview.pdf`.
- Running a versioned generator intentionally replaces the canonical PDF.
  Recompile `main.tex` immediately afterward so `main.pdf` shows the selected
  design. Switching back is simply a matter of running the other generator and
  recompiling.
- Do not create parallel `-v1.pdf`, `-v2.pdf`, `-v1.npz`, or `-v2.npz` files
  when the alternatives present the same underlying data. The canonical PDF
  and its same-stem data twin remain one-to-one; only the rendering code is
  versioned.
- Treat an archived generator as frozen. A newer renderer may reuse its helper
  code only if doing so cannot change the archived version's output; otherwise
  move the new behavior into the newer renderer.

## Design at final physical size

- Render each figure at the width at which LaTeX will place it. The current
  ICLR full text width is `5.5 in`; do not render on a large canvas and depend
  on LaTeX scaling to make it fit.
- Choose height from the content rather than forcing a uniform aspect ratio.
  Remove unused outer margins, but preserve enough room for titles, labels,
  annotations, and token boxes.
- Use vector PDF for the paper. Keep the same physical width across comparable
  figures so equal point sizes remain visually equal in the compiled paper.
- Use an optional thin bounding box while debugging layout. It should reveal
  wasted space and clipping, not become part of the design.

## Typography

- Use the paper serif stack: Times New Roman, Times, then DejaVu Serif.
- At a `5.5 in` full-width figure, use this baseline unless a figure has a
  specific reason to differ:

  - body text and axis labels: `8 pt`;
  - tick labels, legends, and panel titles: `7 pt`;
  - compact token pills: `4 pt` monospace;
  - secondary in-panel annotation: about `6 pt`.

- Use concise, bold, left-aligned panel titles. Align titles in the same row to
  the same physical baseline, even when their axes have different heights.
- Treat the baseline font sizes as defaults, not global mandates. Reduce tick
  label size locally when the labels have a large visual footprint—for example,
  when an axis contains several multi-digit values or values above 99. Keep
  neighboring figures at the baseline size when their shorter labels are
  already clear; do not turn a local crowding fix into a paper-wide font change.
- Prefer compact mathematical axis labels such as $u_1$ and $u_2$ when the
  coordinate system is already explained in the caption. Place paired labels
  consistently and keep them close to their axes.
- Labels must name the actual quantity plotted. For example, use
  “Logcosh contrast” for the FastICA objective rather than an ambiguous generic
  label.

## Multi-panel layout

- Comparable panels should share limits, ticks, binning, and normalization.
- When panels share a y-axis, show the y label and tick numbers only on the
  left panel. Retain tick marks on the other panel so the shared scale remains
  legible.
- Remove duplicated legends. Place a shared legend where it cannot collide
  with either panel's axes or title.
- For figures with two independent encodings, factorize the legend instead of
  enumerating every combination. For example, use color and marker for ICA
  sign orientation, and line style for fitting budget.
- Prefer deliberate figure and axes coordinates for dense composite figures;
  automatic layout often introduces inconsistent gaps or excess whitespace.
- Use fixed physical-point spacing for compact text boxes. Their gap must not
  depend on token width, axis range, or data coordinates.

## Color and visual semantics

- Follow the shared method palette in `plot-style-policy.md` for quantitative
  comparisons.
- Within a conceptual illustration, keep one semantic color per direction:
  ICA is blue; the random direction is muted red; background activations and
  reference fits are gray.
- Use white-filled, colored-outline markers for highlighted projections so the
  direction line remains legible. Draw direction lines underneath markers.
- Use subtle background points and stronger highlights. In Figure 1, ordinary
  background activations use `s=8`; all highlighted activations and projected
  points use `s=15`.
- Dashed lines should communicate a specific reference or construction, such
  as a Gaussian fit, a top-$k$ cutoff, or an orthogonal projection—not merely
  add visual variety.
- Related conditions should share a color family. In ICA comparisons, pale
  blue denotes the arbitrary-sign condition and strong blue denotes the
  stronger-tail-sign condition; SAE uses muted red.
- Make draw order explicit when curves overlap. Do not rely on incidental loop
  order or a common `zorder`. Matplotlib compound artists such as `errorbar`
  may internally offset their child artists' z-orders.
- In method-comparison figures, order legends as ICA, SAE, then other
  baselines. Draw curves in the reverse priority order: other baselines first,
  SAE next, and ICA last, so the paper's primary method remains visible.
- A reproduction figure may preserve the reproduced method first in its legend
  when that ordering is part of the figure's scientific narrative; document
  such exceptions explicitly. Figure 4 is the current SAE-reproduction
  exception.
- A corrective overlay should not receive global visual priority unless that
  priority is meaningful. If a hollow marker belongs between two curve layers,
  assign explicit sequential z-orders to the base curve, overlay, and later
  curves.

## Quantitative curves and incomplete evaluations

- Directly comparable experiments should reuse panel dimensions, axis limits,
  method colors, marker shapes, line widths, and uncertainty styling. This is
  especially important when a robustness figure is intended to be compared
  mentally with a main-text result.
- Show uncertainty without dominating the curves: use thin error bars, small
  caps, and markers large enough to remain legible at final physical size.
- Use a hollow version of the ordinary method marker to flag a point computed
  from fewer observations than the nominal budget. Do not annotate every
  complete point.
- State the hollow-marker convention and exact exceptional sample counts in
  the caption or nearby prose. A hollow marker alone is ambiguous.
- Draw the hollow marker immediately after its method's base curve, then draw
  later methods in the intended order. Verify the result in the rendered PDF,
  because a later overlapping curve may legitimately cover part of it.

## Distribution and ranked-token panels

- Fit a background reference distribution using background samples only. Do
  not let deliberately highlighted related tokens influence the Gaussian fit;
  this reflects the regime in which the background population is much larger
  than the exceptional group.
- Use identical x limits and y limits for directly compared distributions.
- If token pills summarize ranked points, their left-to-right order must match
  the plot axis: smaller projection on the left, larger projection on the
  right.
- State the population over which “top-$k$” is computed. In Figure 1, the five
  rugs and token pills are the top five projections among all 305 displayed
  points (300 sampled background activations plus 5 related activations).
- An annotation arrow should point to the rug corresponding to the intended
  ranked item. Figure 1 uses the second-largest projection to avoid crowding at
  the extreme edge.
- Token pills representing the same kind of object should use identical font
  size, padding, border width, and fixed inter-token spacing. Color may encode
  the associated direction.

## Captions and claims

- A figure should be understandable at final paper scale without relying on
  color alone; the caption supplies the precise sampling and aggregation
  details.
- When reproducing a published comparison, identify the source paper in the
  caption. State the controlled intervention in reader-facing terms rather
  than internal names such as `fixed_ica`.
- Captions must distinguish illustrative choices from measured claims. For a
  conceptual figure, identify the model, layer, sample sizes, selection rule,
  coordinate transformation, and reference-fit population when these affect
  interpretation.
- Do not claim that a hand-selected example is representative. Use it to
  explain the mechanism, then support the general claim with aggregate
  experiments elsewhere in the paper.
- Keep plot-specific visual choices in the paper renderer and scientific values
  in the same-stem data twin. This lets styling change without silently changing
  the underlying evidence.

## Final check

Before accepting a paper figure:

1. Render it at its intended physical width and inspect it inside `main.pdf`.
2. Check font sizes and title baselines against neighboring paper figures.
3. Check shared axes, ordering, legend placement, and annotation targets.
4. Confirm that the caption agrees with the data preparation and rendering
   code.
5. Regenerate without debug borders or other diagnostic decorations.
6. Compile the whole paper and verify float order. A figure must not appear
   before its appendix section heading or drift into the following section;
   use an explicit float boundary when needed.
7. Confirm that filenames, script names, CLI targets, and LaTeX labels remain
   semantically correct if another figure is inserted earlier in the paper.

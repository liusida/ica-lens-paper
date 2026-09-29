# Plot style policy

Use this policy for paper figures and experiment previews. A figure should make
the scientific comparison legible without implying measurements that were not
performed.

## Visual identity

- Use a serif paper font: Times New Roman, Times, then DejaVu Serif as fallback.
- Use the same method encoding everywhere:

  | Method | Color | Marker |
  | --- | --- | --- |
  | ICA | `#3D5F99` | diamond (`D`) |
  | SAE | `#B45F4D` | circle (`o`) |
  | PCA | `#5B8C6A` | triangle (`^`) |
  | Random | `#777777` | square (`s`) |

- Use solid lines for primary results and dashed lines only for a clearly
  identified control or variant.
- Draw methods back-to-front as Random, PCA, SAE, ICA so ICA remains visible.
  Order the legend independently as ICA, SAE, PCA, Random; place a variant next
  to its parent method.
- Use approximately 1.5 pt lines and 3–4 pt markers. Use an unframed, shared
  legend above multi-panel figures.
- Remove top and right spines. Show light horizontal grid lines only.
- Keep grid lines behind data, around gray `0.88–0.90` and `0.45–0.6` pt. Use
  thin, consistent axes (about `0.7` pt).

## Paper sizing and typography

- Build figures at their intended publication size. Use about `3.35` inches
  for one column and `6.9` inches for two columns; do not create an oversized
  canvas and rely on the paper renderer to shrink it.
- A useful two-column baseline is 8 pt for titles, labels, and body text, about
  7 pt for tick labels, and about 7–7.5 pt for legends. Keep this hierarchy
  proportional when another base size is required.
- Apply figure styles in a local Matplotlib `rc_context` so one plot does not
  silently alter later plots in a notebook or process. Embed TrueType fonts in
  both PDF and PostScript (`pdf.fonttype = 42`, `ps.fonttype = 42`).

## Layout and labels

- Design aggregate paper figures for a single-column or two-column paper width;
  keep panels compact and share axes when their scales and semantics match.
- Use bold, concise panel titles with stable model/dataset names.
- Prefer left-aligned semibold or bold panel titles in dense multi-panel figures; they
  are easier to scan than centered titles. Put layer counts, sample counts, or
  other qualifiers in a smaller in-panel annotation or subtitle rather than
  lengthening the primary title.
- Put the y-axis label only on the leftmost panel when axes are shared. Use one
  shared x-axis label when practical.
- Prefer sentence case for axis labels. Name the measured quantity and unit or
  normalization explicitly; avoid unexplained abbreviations.
- Use log-scaled x-axes for feature/direction budgets spanning orders of
  magnitude. Label only values actually evaluated, plus a measured native or
  full-basis endpoint. Rotate crowded tick labels 45 degrees.
- Do not annotate points unless the annotation communicates essential status
  such as a partial run. Explain special markers and controls in the caption.
- When panels compare distributions, use the same axis limits, binning,
  normalization, and color scale across panels. For nested uncertainty or
  percentile bands, make outer bands lighter and inner bands darker, then draw
  the summary curve above them.
- Reserve space explicitly for shared legends and labels. For dense paper
  figures, deliberate `subplots_adjust` margins are often more stable than
  relying on `tight_layout` to resolve an outside legend.

## Scientific integrity

- Never interpolate, extrapolate, or display a requested budget as though it
  were an observed effective budget.
- Aggregate only comparable, completed observations and state what is averaged.
  Mark partial panels explicitly and report which layers are included.
- Plot SAE points at measured mean active features. Collapse repeated budgets
  that produce the same native sparse reconstruction.
- Use a star only for an evaluated endpoint: complete basis for linear methods
  or native sparsity for an SAE. Do not also draw the ordinary method marker at
  that endpoint.
- A line may connect the largest evaluated linear budget to its evaluated
  full-basis endpoint, but the caption must say so. Do not invent intermediate
  points.
- Keep legend order, drawing order, aggregation, tick selection, and endpoint
  semantics explicit in code rather than relying on plotting-library defaults.

## Outputs and reproducibility

- Official aggregate figures: save both PNG and PDF. Use at least 240 dpi for
  PNG and embed TrueType fonts in PDF (`pdf.fonttype = 42`).
- Preview and diagnostic figures: PNG only unless a vector copy is needed.
- Save a same-stem `.txt` caption beside every official figure. It should state
  the metric, aggregation, included methods, endpoint meaning, and any controls
  or incomplete coverage.
- Generate figures from committed result files through a checked-in experiment
  script. Regeneration must not rerun measurement or silently overwrite outputs;
  require an explicit `--force` flag.
- Close figures after saving and use a writable temporary Matplotlib cache so
  headless runs are deterministic and do not pollute the repository.

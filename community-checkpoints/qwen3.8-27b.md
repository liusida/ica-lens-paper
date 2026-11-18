# Qwen3.8-27B Bilingual ICA Lens (All 64 Layers + Bundled J-Lens Readouts)

> **Contributor note:** This checkpoint introduction was written by lmxxf. It describes the contributor’s experiments, findings, and interpretations of the Qwen3.8-27B ICA Lens.

- **Hugging Face Checkpoint**: [`lmxxf/Qwen3.8-27B-ica-lens`](https://huggingface.co/lmxxf/Qwen3.8-27B-ica-lens) (`v3` on `main`; legacy snapshots preserved at tags `v1.0` and `v2.0`)
- **Companion Bilingual Jacobian Lens (`N=100`)**: [`lmxxf/Qwen3.8-27B-jacobian-lens`](https://huggingface.co/lmxxf/Qwen3.8-27B-jacobian-lens)
- **Base Model**: `Qwen/Qwen3.8-27B-FP8` (dequantized blockwise to `BF16`; residual stream block outputs `0..63`, hidden size `d = 5,120`)
- **Scale & Format**: `64` layers × `5,120` signed non-overcomplete components = **`327,680` directions**, packaged in native v4 format for `icalens >= 0.4.1` (`15.8 GB` total)
- **Discussion & Convergence Audit**: [`liusida/ica-lens-paper#4`](https://github.com/liusida/ica-lens-paper/issues/4)

---

## 1. Overview

This community checkpoint provides a full-depth, 1:1 English/Chinese **Independent Component Analysis Lens (ICA Lens)** across all 64 transformer blocks of `Qwen3.8-27B`.

Every component profile (`component_profiles/resid_post/layer_00` .. `layer_63`) comes pre-populated with:
1. **Held-out fourth-moment statistics** evaluated on **`1,365,856` tokens** across **`1,200` disjoint full-structure articles** (`682,928` EN + `682,928` ZH);
2. **Real top-10 causal activation contexts** drawn from unseen bilingual documents;
3. **Standard Logit Lens** vocabulary projections;
4. **Bundled bilingual `J Lens` readouts** (`additional_readouts["J Lens"]`) computed via an `N=100` bilingual Jacobian Lens so early- and mid-layer directions remain directly readable in vocabulary space without suffering 40-layer unembedding distortion—and without forcing users to download the separate `3.1 GB` raw Jacobian tensor.

---

## 2. Quickstart (`icalens >= 0.4.1`)

```python
import numpy as np
from icalens import ICALens

lens = ICALens.from_pretrained("lmxxf/Qwen3.8-27B-ica-lens")
assert lens.available_layers == tuple(range(64))

# Project BF16/FP32 residual activations h of shape (..., 5120) onto Layer 32 ICA axes:
# scores = (h / ||h|| - center) @ reading_matrix.T
rng = np.random.default_rng(0)
h = rng.standard_normal((4, 5120)).astype("float32")
scores = lens.transform(h, layer=32)
unit_recon = lens.inverse_transform(scores, layer=32)

# Inspect Layer 32 Component 1909 (including precomputed bilingual J-Lens readouts)
profile = lens.component_profile(layer=32, component=1909)
print("Held-out excess kurtosis:", round(profile["score_statistics"]["excess_kurtosis"], 2))  # 15.71
print("Bundled readouts:", list(profile.get("additional_readouts", {}).keys()))  # ['J Lens']
j_top = profile["additional_readouts"]["J Lens"]["positive"]["top_tokens"]
print("Top J-Lens tokens:", [t["text"] for t in j_top[:6]])
# ['…”', '查看答案解析', '点击上方蓝字', '...”', 'EMPLARY', '企业相册']
```

You can also browse components interactively in Jupyter Notebook using `icalens`'s built-in profile and text-analysis widgets.

---

## 3. Fitting Lessons at 27B Scale (`v1 → v2 → v3`)

Fitting 5,120-dimensional FastICA cleanly across all 64 layers of a 27B model exposed two subtle data-sampling pitfalls that do not show up on Fitting-set kurtosis alone. Tracking the **Fit-to-Held-Out Kurtosis Ratio** across three iterations resolved both:

![v3 64-layer fit vs held-out convergence](https://huggingface.co/lmxxf/Qwen3.8-27B-ica-lens/resolve/main/reproducibility/assets/v3-64layers-convergence.svg)

1. **Document diversity matters far more than raw token count (`v1 → v2`)**:
   - In `v1`, `200,704` tokens were sliced contiguously from long texts, inadvertently spanning only **`66` documents** (and evaluated on `17` documents). While shallow and final layers generalized well, middle discourse layers (`L24–L48`) memorized book-specific proper nouns—causing `L32` median excess kurtosis `κ` to collapse **`55.7×`** from `55.10` (fit) to **`0.99`** (held-out).
   - Sampling `1,000,000` non-initial tokens across **`1,829` distinct documents** in `v2` immediately eliminated all `872` overfitted mid-layer spikes (`0 / 872` reproduced at cosine `> 0.8`) and raised `L32` held-out `κ` tenfold to `10.02`.
2. **Truncating to opening 512 tokens starves discourse-arc & synthesis axes (`v2 → v3`)**:
   - Even in `v2`, middle layers still showed a `2.1×–2.6×` fit/test gap because truncating every article to its first `512` tokens destroyed **`introduction → body development → concluding synthesis`** structure.
   - In **`v3`**, we adopted a **`4,096`-token full-article budget**: articles with `≤ 4,096` tokens (`~2/3` of Wikipedia/fine-web longform texts) are forwarded intact; longer articles use **`first 1,024 + explicit omission marker + central 2,048 + omission marker + final 1,024`**, with position `0` (attention sinks), omission markers, and held-out literary evaluation corpora (*Dream of the Red Chamber*) strictly masked out before sampling `≤ 512` positions per document across **`8,000` training articles (`2.5M` fit tokens, `18.26M` forwarded)** and **`1,200` disjoint test articles (`1.37M` test tokens)**.

### Three-Generation Convergence Summary

| Version | Fit Corpus | Held-Out Test Corpus | 64-Layer Fit/Test Ratio | `L32` Held-Out Median `κ` | Global Held-Out Median `κ` (`327,680` axes) | Strong Non-Gaussian Axes (`κ > 10`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`v1`** (`v1.0`) | `200.7K` tokens (`66` contiguous docs) | `51.2K` tokens (`17` docs) | `2.89× – 56.00×` | `0.99` | `4.19` (`9.74` on 500-doc) | `162,158` (`49.5%`) |
| **`v2`** (`v2.0`) | `1.00M` tokens (`1,829` docs, head-512) | `255.5K` tokens (`500` docs, head-512) | `1.18× – 2.61×` | `10.02` (`9.03` on full-doc test) | `30.10` (`29.93` on full-doc test) | `237,316` (`72.42%`)* |
| **`v3` (`main`)** | **`2.50M` tokens (`8,000` full-arc docs)** | **`1.37M` tokens (`1,200` full-arc docs)** | **`1.07× – 1.48×`** (`0` layers `>1.5×`) | **`16.80`** (**`1.86×`** vs same-test `v2`) | **`41.77`** (**`+39.6%`** vs same-test `v2`) | **`268,320` (`81.88%`, `+29,133` vs same-test `v2`)** |

*\*Evaluated on the exact same `1,365,856` full-article test tokens, `v2` yields `239,187` (`72.99%`) axes with `κ > 10`, whereas `v3` yields `268,320` (`81.88%`). Conversely, when `v3` is evaluated back on `v2`'s truncated head-512 test set, both score ~`29–30`, confirming that the `+29,133` newly resolved non-Gaussian axes specifically capture mid-article development and concluding discourse directions.*

Fitting ran in **22.96 hours** across two NVIDIA DGX Spark nodes (`GB10`, `2 × 128 GB` unified memory) using `FP64` covariance accumulation, `50` fixed FastICA (`logcosh`) iterations, positive-skew sign orientation, and strict duality verification (`max |C @ M - I| <= 8.81e-5`).

---

## 4. Selected Case Studies on `Qwen3.8-27B` `v3`

Using this `v3` checkpoint alongside its bundled bilingual `J Lens` readouts, we conducted four controlled mechanistic studies (each linked directly to its standalone experiment folder and technical write-up):

### 4.1 Orthographic Expectation Violation vs. Semantic Polysemy (`L40–L50`)
When probing polysemous concepts (`apple` vs. `苹果` across fruit and tech-company contexts), English and Chinese blind top-1 selections land on the **exact same axis across 12 consecutive middle layers (`L24–L35`)**. Between `L40` and `L50`, however, Chinese continues rotating cleanly among food, commercial-brand, and tech-R&D axes, whereas lowercase English `apple` in tech sentences (`"... iPhone design ... mentioned apple"`) is dominated for ten straight layers by single-sided positive axes such as `L40 C776` and `L45 C582`. Inspecting real corpus exemplars (`British royal`, `National cash`, `Coaches vs. cancer`) and `J Lens` tokens (`microsoft`, `firefox`, `whatsapp`, `america`, `texas`, `bible`) revealed a dedicated **"expected-capitalized proper noun written in lowercase"** detector rather than a semantic split.

- *Reproducibility & write-up*: [`reproducibility/` (in HF repo)](https://huggingface.co/lmxxf/Qwen3.8-27B-ica-lens/tree/main/reproducibility) · [`wechat-369-icalens`](https://github.com/lmxxf/grocery-haven/tree/master/wechat-369-icalens) · [`wechat/372.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/372.md)

### 4.2 Cross-Lingual Kinship Geometry & Lexical Slots (`叔叔/舅舅` vs. `uncle`)
Chinese obligatorily distinguishes paternal lineage (*in-group*) from maternal/affinal relatives (*out-group*), whereas English collapses both into `uncle` / `aunt` / `cousin`. Contrasting 100% consanguineous same-generation pairs (`伯父/叔叔/姑姑` vs. `大舅/舅舅/姨妈`) uncovered a clean three-stage hierarchy:
- **`L13 C2665` (`κ = 60.04`)**: Unifies physical spatial exterior (`室外/门外`), social exterior (`外人/外行`), and maternal kinship (`+5.22σ` even on single-token `姥爷/姥姥` vs. `爷爷/奶奶` containing no literal `"外"` character; `10/10`). *(Notably, vanilla Logit Lens degrades to `['对于']` across the remaining 50 nonlinear layers, whereas corpus activations + `J Lens` expose the exact `"外"` operator.)*
- **`L25 C3140` (`κ = 74.97`) / `L26 C3526` (`κ = 46.48`) / `L28 C544` (`κ = 114.0`)**: Single-pole **affinal/marriage-link operators** (`娘家` ranks #1 across the entire vocabulary on `L28 C544`, with top corpus hit `Katherine ( née` at `z = 48.53`), separating zero-shared-character Chinese test sentences at **`60/60`** accuracy while leaving paternal relatives near baseline.
- **`L42 C1992` (`κ = 34.30`, ~2/3 model depth)**: Paternal bloodline hugs zero (`+0.20`) while maternal/affinal side activates (`+2.41`, `60/60` word-mean accuracy).
- **Crucial Cross-Lingual Negative Evidence**: When given explicit English (`"my father's brother ... uncle"` vs. `"my mother's brother ... uncle"`), **all ten kinship bars flatten into identical twins** (`L25 C3140`: `+10.53` vs. `+10.47`, `Δ = -0.06`). Even under colloquial Chinese context ending with code-switched English `' uncle'`, maternal-vs-paternal `Δz` collapses to **`+0.02`** on `L42 C1992` (rank `4,219 / 5,120`, `5/10` chance). Rather than imposing an innate "cultural nationality," the model invokes lineage operators strictly on demand when required by lexical slots.
- *Code & write-up*: [`icalens-v3-exp1-kinship-20261116`](https://github.com/lmxxf/grocery-haven/tree/master/icalens-v3-exp1-kinship-20261116) · [`wechat/373.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/373.md)

### 4.3 Diachronic Calibration (*Chu Shi Biao*) & Historical Role Geometry (*Dream of the Red Chamber* & Qing Memorials)
Can `ICA Lens` audit classical texts without being confounded by modern semantic drift?
1. **Zeroing the Time Machine**: In Zhuge Liang's 227 AD Shu-Han memorial *Chu Shi Biao* (`先帝不以臣卑鄙`), `"卑鄙"` meant *humble origins/low station*, not *morally despicable*. Across 38 validated moral-pejorative layers (`L10` and `L12–L48`), feeding a 2,369-token pure Classical Chinese context from Chen Shou's *Records of the Three Kingdoms* yields the lowest median modern-pejorative residual (**`17.0%`**, outperforming a 364-word English academic seminar framing prompt at `25.6%`) and drives representative middle-deep axes such as **`L30 C846` down to `-0.43%`** (`L38 C1448`: `-4.8%`; `L45 C1564`: `-1.9%`).
2. **Auditing `"奴才"` (*Nucai*)**: On the pre-filtered 120-chapter *Dream of the Red Chamber* and 10 archival Yongzheng-era palace memorials (including Huguang Governor-General Yang Zongren's 1723 *Greeting Memorial*, National Palace Museum `#022688`, where Emperor Yongzheng crossed out `"奴才"` in vermilion ink and wrote **`"称臣得体"`** — *"Addressing yourself as 'minister' is proper"*):
   - Polite self-referential `"奴才"` sits calmly inside the routine deference band on `L30 C846` (`+1.93` / `+2.02`, nestled between same-sentence `"小的"` at `+1.65` and `"下官"` at `+2.28`), far below scolding/abusive uses (`+7.19 – +10.07`). Across **`L37–L46` (10 consecutive layers)**, the #1 axis separating self-reference from abuse is led without exception by **first-person pronoun agency (`myself / my / 我自己`)** with **`9/9` (`100%`)** held-out chapter accuracy.
   - Replacing `"臣"` (*minister*) with `"奴才"` (*bondservant*) in place at the end of each memorial's authentic causal left context reveals an in-group vs. outer-court see-saw across three stages: early entry shifts from historical outer-court ministers (`L18 C1649` `北宋大臣`, `-11.04`) toward **personal steward/valet (`L12 C2054` `管事 / 总管 / valet`, `+6.70`, `10/10`)**, `Slave` etymology (`L18 C1690`, `+5.78`), and an 18th-century Qing temporal anchor (`L18 C230`, `+6.25`, #1); middle layers (`L25–L32`) drop sharply on **outer-court officialdom (`L25 C1146` `大臣 / 宰相` `-5.70`; `L32 C2829` `Governor` `-4.25`; `L32 C3016` `御史 / chancellor` `-3.97`)** while rising **`10/10` on household/kinship axes (`L25 C2783` `family / 家庭` `+3.40`; `L26 C2910` `家庭成员` `+3.27`; `L28 C2167` `家族 / lineage` `+3.91`, all #3 word-mean)**; and deep layers (`L45 C1194` `+7.27`, #1; `L53 C1681` `+6.69`, #1 `酌 / 办理 / 督办 / 朕`) converge onto Qing memorial-rescript bureaucratic register.
- *Code & write-up*: [`icalens-v3-exp1b-archaic-nucai-20261116`](https://github.com/lmxxf/grocery-haven/tree/master/icalens-v3-exp1b-archaic-nucai-20261116) · [`wechat/374.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/374.md)

### 4.4 Non-Gaussian Continents & Chain-of-Thought Navigation (+ Interactive 3D Atlas)
On the 5,120-dimensional unit sphere, isotropic Gaussian vectors (`κ ≈ 0`) form a featureless ocean (`4,096` random Gaussian vectors projected onto `L32` yield `99%` quantiles `z_max+ = 4.64` and spatial kurtosis `Kurt_d = 0.16`), whereas the `268,320` heavy-tailed axes (`κ > 10`) act as labeled semantic islands. Combining spherical geodesic curvature (`s_t`, `κ_g`, 5-step local tangent effective rank `rank5`) with live KV-cache `ICA Lens` readouts across 44 generation traces revealed:
- **High-`κ` Ridges vs. Low-`κ` Operator Shallows**: Neither memorized poetry (`z_max+ = 18.97, Kurt_d = 15.12`) nor parenthetical multi-step arithmetic (`z_max+ = 10.68, Kurt_d = 2.99, ΔC_10 = -4.50%`) ever drops into the Gaussian ocean (`0%` steps). During arithmetic, energy shifts from `κ > 10` encyclopedic noun islands into `1 < κ < 10` syntactic/equational channels (e.g., `L32 C3416`, `κ = 1.2`, hitting `z = 13.93` on intermediate digits).
- **Corpus Imprint Caught Red-Handed**: Tracing `"明月"` (*bright moon*) in Li Bai's *Quiet Night Thought* vertically across all 64 layers shows `L20 C1277` (`z = 34.12`) lighting up `"月球 / 嫦娥 / lunar"`—yet at discourse hub **`L32 C1909` (`z = 16.67`)**, the #1 dominant island across the entire poem reads **`"查看答案解析 / 点击上方蓝字"` (*"Click blue link above to view exam answer key"*)**, explaining why the base model immediately appends a middle-school exam commentary right after finishing the final verse.
- **`127.4°` Self-Correction Island Relay**: When catching a planted `49×51 = 2501` error and emitting `"等等，让我重新算："` (*"Wait, let me recalculate:"*), the comma executes a **`127.4°` (`2.224 rad`, `κ_g = 3.53`)** obtuse geodesic turn on the sphere while handing off cleanly across functional islands: `L32 C2165` (anomaly exclamation) → `L32 C1542` (`actually / 并非`) → `L32 C626` (`myself / 我的`) → `L32 C2436` (`重新 / rebuild`, `z = 11.54`) → `L32 C1444` (`therefore`), with **zero Top-5 island overlap** before vs. after correction.
- **Falsifying the Low-Rank Limit-Cycle Hypothesis (Geometry of Overthinking)**: After answering `23+19 = 42` in 3 tokens, blocking `EOS` for 96 steps does *not* trap the 27B model in a low-rank repetition loop (`rank5 = 3.43` in positive repetition controls). Instead, `rank5` **rises from `4.05` to `4.22`**, entropy jumps `10×` (`0.14 → 1.47 bits`), and the trajectory traverses **`53` distinct primary islands** to hallucinate an alternative exam-typo hypothesis: *"(Note: if the problem meant 23×19, the result would be 437...)"*.
- **Interactive 3D WebGL Atlas** (built from measured `v3` residuals & island readouts): [**Non-Gaussian Earth: 3D CoT Navigation Atlas**](https://lmxxf.github.io/DLSS5-Explanation/AIDemo/NonGaussianEarth/)
- *Code & write-up*: [`icalens-v3-exp2-cot-navigation-20261117`](https://github.com/lmxxf/grocery-haven/tree/master/icalens-v3-exp2-cot-navigation-20261117) · [`wechat/375.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/375.md)

---

## 5. Direct Links to Checkpoints, Experiment Folders & Write-Ups

- **Model Checkpoints**:
  - ICA Lens `v3` (all 64 layers + bundled `J Lens` profiles): [`lmxxf/Qwen3.8-27B-ica-lens`](https://huggingface.co/lmxxf/Qwen3.8-27B-ica-lens)
  - Companion `N=100` Bilingual Jacobian Lens: [`lmxxf/Qwen3.8-27B-jacobian-lens`](https://huggingface.co/lmxxf/Qwen3.8-27B-jacobian-lens)
- **Foundational Lens Fitting & Diagnostics (Code + Chinese Notes)**:
  - Bilingual `J-Lens` fitting & steering: [`wechat-365-jlens-fit`](https://github.com/lmxxf/grocery-haven/tree/master/wechat-365-jlens-fit) ([`365.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/365.md)) · [`wechat-366-jlens-experiments`](https://github.com/lmxxf/grocery-haven/tree/master/wechat-366-jlens-experiments) ([`366.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/366.md))
  - Fourth-moment kurtosis theory, `v1→v2→v3` convergence & polysemy probes: [`wechat-367`](https://github.com/lmxxf/grocery-haven/tree/master/wechat-367) ([`367.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/367.md)) · [`wechat-369-icalens`](https://github.com/lmxxf/grocery-haven/tree/master/wechat-369-icalens) ([`369.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/369.md)) · [`372.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/372.md)
- **Downstream `v3` Mechanistic Studies (Code + `DevHistory.md` + Chinese Notes)**:
  - Cross-lingual kinship (`叔/舅` vs. `uncle`): [`icalens-v3-exp1-kinship-20261116`](https://github.com/lmxxf/grocery-haven/tree/master/icalens-v3-exp1-kinship-20261116) · [`373.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/373.md)
  - Diachronic calibration (*Chu Shi Biao*) & Qing `"奴才"` vs. `"臣"`: [`icalens-v3-exp1b-archaic-nucai-20261116`](https://github.com/lmxxf/grocery-haven/tree/master/icalens-v3-exp1b-archaic-nucai-20261116) · [`374.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/374.md)
  - Non-Gaussian CoT navigation & spherical curvature: [`icalens-v3-exp2-cot-navigation-20261117`](https://github.com/lmxxf/grocery-haven/tree/master/icalens-v3-exp2-cot-navigation-20261117) · [`wechat-363`](https://github.com/lmxxf/grocery-haven/tree/master/wechat-363) · [`375.md`](https://github.com/lmxxf/ai-theorys-study/blob/main/wechat/375.md) · [Interactive 3D WebGL Atlas](https://lmxxf.github.io/DLSS5-Explanation/AIDemo/NonGaussianEarth/)

# Python API

本包的公开接口包括 `ICALens`、`CaptureResult`、`AnalysisResult` 以及异常类型。

```python
from icalens import ICALens
```

## 创建或加载 Lens

### `ICALens(...)`

```python
lens = ICALens(
    model_id="openai-community/gpt2",
    model_type="base",
    icalens_preprocessing="none",
)
```

```python
ICALens(
    *,
    model_id=None,
    model_revision=None,
    model_type="base",
    activation_site="resid_post",
    layer_indexing="hidden_states",
    row_normalize=None,
    icalens_preprocessing=None,
    norm_eps=1e-12,
)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `model_id` | `str`，必填 | 待分析语言模型的 Hugging Face 仓库 |
| `model_revision` | `str \| None = None` | 语言模型的精确版本；若省略，则在首次捕获时解析 |
| `model_type` | `"base" \| "instruct" = "base"` | 基础模型或指令/聊天模型 |
| `activation_site` | `str = "resid_post"` | 激活位置名称 |
| `layer_indexing` | `str = "hidden_states"` | 产物中记录的层编号约定 |
| `icalens_preprocessing` | `"none" \| "l2" \| "geometric-median-l2" \| None = None` | 标准 FastICA 中心化与白化之前使用的变换；`None` 保留旧版的 L2 默认行为 |
| `row_normalize` | `bool \| None = None` | 兼容旧代码的参数；新代码应使用 `icalens_preprocessing` |
| `norm_eps` | `float = 1e-12` | 归一化数值下限 |
| **返回** | `ICALens` | 尚未拟合的新 Lens |

`icalens_preprocessing="none"` 将原始激活直接交给标准 FastICA 做中心化和白化；
`"l2"` 会先把每个 token 激活归一化到单位长度；`"geometric-median-l2"`
则先减去稳健中心，再做 L2 归一化。所选模式会写入产物，并由 `transform()`、
`inverse_transform()` 和 `analyze()` 自动复用。

### `ICALens.from_pretrained(...)`

```python
lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `repo_id_or_path` | `str \| Path`，必填 | Hub Model 仓库或本地 ICA Lens 目录 |
| `revision` | `str \| None = None` | ICA Lens 仓库的分支、tag 或 commit |
| `cache_dir` | `str \| Path \| None = None` | Hugging Face 缓存目录 |
| `token` | `str \| bool \| None = None` | Hugging Face 身份认证设置 |
| `local_files_only` | `bool = False` | 仅使用本地缓存 |
| `force_download` | `bool = False` | 强制重新下载 |
| **返回** | `ICALens` | 根据 manifest 初始化的 Lens |

从 Hub 加载时，程序最初只下载 `icalens.json`；每一层的张量文件会在首次使用该层时
按需下载。这里的 `revision` 指 ICA Lens 仓库版本，而 `model_revision` 指待分析语言
模型的版本。

## 分析模型输入

### `analyze(...)`

```python
result = lens.analyze("She deposited the check.", layer=6)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `inputs` | `str \| list[dict[str, str]]`，必填 | 纯文本或完整 `{role, content}` 对话 |
| `layer` | `int`，必填 | 从 0 开始的 transformer block 编号 |
| `model` | `torch.nn.Module \| None = None` | 可选的外部语言模型 |
| `tokenizer` | tokenizer 或 `None` | 可选的外部 tokenizer |
| `token_scope` | `"all" \| "content" \| "user" \| "assistant" = "all"` | 对话中返回的位置 |
| `context_length` | `int \| None = None` | 最大编码长度 |
| `device` | `str \| torch.device \| None = "auto"` | 模型设备；自动模式优先 CUDA |
| `verbose` | `bool = False` | 输出模型加载、激活捕获和分数计算各阶段的耗时进度 |
| **返回** | `AnalysisResult` | token、激活、ICA 分数和能量占比 |

### `generate(...)`

生成续写，并可选择对带符号的 ICA 坐标增加偏移或进行钳制：

```python
baseline = lens.generate(messages, max_new_tokens=16)

steered = lens.generate(
    messages,
    layer=5,
    steer=(188, -12.0),
    max_new_tokens=16,
)
```

```python
lens.generate(
    prompt,
    *,
    layer=None,
    clamp=None,
    steer=None,
    steering_scope="current-position",
    max_new_tokens=64,
    device="auto",
    model=None,
    tokenizer=None,
    **generation_kwargs,
) -> str
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `prompt` | `str \| list[dict[str, str]]`，必填 | 原始提示词或聊天消息；消息会自动套用聊天模板 |
| `layer` | `int \| None = None` | 要修改的残差流层；使用 `clamp` 或 `steer` 时必填 |
| `clamp` | `tuple[int, float] \| Mapping[int, float] \| None = None` | 一个 `(成分, 目标分数)`，或多个同时钳制目标的映射；应用于每个处理位置和每个生成步骤 |
| `steer` | `tuple[int, float] \| Mapping[int, float] \| None = None` | 一个 `(成分, 分数偏移)`，或多个同时施加的加性偏移；不能与 `clamp` 同时使用 |
| `steering_scope` | `"current-position" \| "all-positions" = "current-position"` | 使用 `steer` 时，只修改运行时当前位置，或修改预填充与解码期间处理的所有位置 |
| `max_new_tokens` | `int = 64` | 最多生成多少个续写 token |
| `device` | `str \| torch.device \| None = "auto"` | 模型设备；自动模式优先 CUDA |
| `model` | `torch.nn.Module \| None = None` | 可选的外部语言模型 |
| `tokenizer` | tokenizer 或 `None` | 可选的外部 tokenizer |
| `**generation_kwargs` | 关键字参数 | 继续传给 `model.generate()` 的参数 |
| **返回** | `str` | 只包含续写、不包含提示词的解码文本 |

不提供 `clamp` 或 `steer` 时，这是普通生成。`clamp` 会替换绝对分数并恢复激活范数；
`steer` 会把 `分数偏移 * 写入方向` 直接加到残差流中，因此要求 Lens 未做逐行归一化，
且没有预处理中心。默认使用贪心解码；也可传入标准生成参数选择其他策略。语言模型会
按需加载，并由同一个 Lens 的后续调用复用。

### `capture(...)`

```python
captured = lens.capture("She deposited the check.", layer=6)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `inputs` | 文本或消息列表，必填 | 模型输入 |
| `layer` | `int`，必填 | 要捕获的 block |
| `model`, `tokenizer` | `None` | 可选的外部模型和 tokenizer |
| `token_scope` | `str = "all"` | 对话返回位置 |
| `context_length` | `int \| None = None` | 最大编码长度 |
| `device` | `str \| torch.device \| None = "auto"` | 模型设备 |
| **返回** | `CaptureResult` | token、位置、ID 和对齐激活 |

### `unload_model()`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| **返回** | `None` | 释放 `capture()`、`analyze()` 或 `generate()` 缓存的模型和 tokenizer |

此方法不会卸载 ICA 层矩阵。

## 为拟合成分建立画像

### `profile_components(...)`

```python
profile = lens.profile_components(
    texts_or_conversations,
    layer=5,
    max_tokens=100000,
    top_k_examples=20,
    provenance={"dataset": {"repo_id": "owner/dataset", "split": "train"}},
    device="auto",
    progress=True,
)
```

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `inputs` | 文本或对话的可迭代对象，必填 | 以流式方式送入模型的输入 |
| `layer` | `int`，必填 | 要建立画像的已拟合层 |
| `token_scope` | `str = "all"` | 对话中纳入统计的位置 |
| `max_tokens` | `int \| None = 100000` | 最多统计的 token 数 |
| `top_k_examples` | `int = 20` | 在偏度所选尾部保留的最大绝对分数样例数 |
| `logit_lens_top_k` | `int = 20` | 每个方向保留的最高和最低词表条目数 |
| `logit_lens_batch_size` | `int = 64` | 同时反嵌入的写入方向数；调低可减少峰值显存 |
| `r_lens` | `str \| Path \| dict \| None = None` | 用于加入后续层近似词表读出的兼容 R-lens 产物 |
| `r_lens_top_k` | `int = 20` | 每个方向保留的 R-lens 词表条目数 |
| `r_lens_batch_size` | `int = 8` | 同时处理的 R-lens 方向数 |
| `provenance` | `dict \| None = None` | 可 JSON 序列化的画像来源信息 |
| `context_length` | `int \| None = 1024` | 每条输入的最大编码长度 |
| `device` | `str \| torch.device \| None = "auto"` | 语言模型设备 |
| `progress` | `bool = False` | 是否显示进度 |
| **返回** | `dict` | 完整逐层画像，同时附加到当前 Lens |

这是拟合后的操作，不会修改中心、读取矩阵或写入矩阵。返回的逐层画像包含 `n_tokens`、
`n_inputs`、画像 `selection`、`provenance` 和 `components` 列表。每个成分条目包含：

| 字段 | 内容 |
| --- | --- |
| `component` | 成分编号 |
| `tail_direction` | 由总体偏度符号选择的尾部方向 |
| `dominant_sign` | `tail_direction` 的兼容性别名 |
| `sign_statistics` | 正负位置比例和正负能量比例 |
| `score_statistics` | 均值、方差、三阶中心矩、偏度、超额峰度及峰度排名 |
| `examples` | 正负两侧保留的高能量样例和 token 计数 |
| `logit_lens` | 正向、负向及主导写入方向对应的词表关联 |
| `r_lens` | 提供兼容 R-lens 时加入的、近似纳入后续层影响的词表关联 |

之后调用 `save()` 可以持久化所有已附加画像。处理多个层时，应在每层完成后调用
`checkpoint_component_profile()`，使已经完成的画像在任务中断后仍能保留。

### `refresh_profile_statistics_from_activations(...)`

从之前捕获的激活重新计算分布统计和基于偏度的尾部选择，而无需重建画像的其余内容：

```python
profile = lens.refresh_profile_statistics_from_activations(
    activations,
    layer=6,
    rows=None,
    batch_size=8192,
    provenance={"source": "saved activations"},
    device="auto",
    progress=True,
)
lens.checkpoint_component_profile("icalens-output/my-icalens", layer=6)
```

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `activations` | `torch.Tensor`，必填 | 在拟合位置捕获的二维激活矩阵 |
| `layer` | `int`，必填 | 已有成分画像的拟合层 |
| `rows` | `torch.Tensor \| None = None` | 可选的一维 `int64` 行选择 |
| `batch_size` | `int = 8192` | 每批变换的激活行数 |
| `provenance` | `dict \| None = None` | 可 JSON 序列化的刷新统计来源信息 |
| `device` | `str \| torch.device \| None = "auto"` | 变换设备 |
| `progress` | `bool = False` | 是否显示刷新进度 |
| **返回** | `dict` | 更新后的逐层画像，同时附加到当前 Lens |

该方法把所选激活行分批送入已经拟合的 ICA 变换，更新正负位置比例、平方能量比例、
分数矩、偏度、超额峰度和 `tail_direction`，并把 Logit Lens 与 R-lens 的 dominant
读出指向所选尾部。已有样例和逐符号读出会保留；不会重新捕获语言模型激活，也不会
重新运行 FastICA。对于磁盘上的激活数据集和可逐层恢复的刷新，应使用
`icalens profile refresh-statistics` CLI。

### `add_r_lens_profile(...)`

无需重新遍历画像数据集，即可向已有成分画像加入 R-lens 读出：

```python
profile = lens.add_r_lens_profile(
    layer=6,
    r_lens="local-r-lens-models/model/lens.pt",
    top_k=20,
    batch_size=8,
    device="auto",
    progress=True,
    allow_base_model_transfer=False,
)
lens.checkpoint_component_profile("icalens-output/my-icalens", layer=6)
```

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `layer` | `int`，必填 | 已有成分画像的层 |
| `r_lens` | `str \| Path \| dict`，必填 | 兼容的已拟合 R-lens 产物 |
| `top_k` | `int = 20` | 每个符号方向保留的词表条目数 |
| `batch_size` | `int = 8` | 同时处理的成分方向数 |
| `device` | `str \| torch.device \| None = "auto"` | 语言模型设备 |
| `progress` | `bool = False` | 是否显示 R-lens 投影进度 |
| `allow_base_model_transfer` | `bool = False` | 显式允许把维度兼容的基座模型 R-lens 复用于指令模型，并记录迁移来源 |
| **返回** | `dict` | 更新后的逐层画像，同时附加到当前 Lens |

该方法保留已有的符号统计、样例和 Logit Lens 条目。R-lens 必须与待分析模型及隐藏维度
匹配，并为请求的 `resid_post` 层提供源层映射。之后调用 `save()` 或
`checkpoint_component_profile()` 持久化更新。
默认要求模型来源完全一致。启用基座到指令模型的复用后，隐藏维度、激活位置和层映射
检查仍然生效，画像中也会保存来源模型与目标模型的信息。

### `component_profile(...)`

```python
component = lens.component_profile(layer=5, component=188)
component  # 在 Jupyter 或 Colab 中显示画像面板
```

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `layer` | `int`，必填 | 已建立画像的层 |
| `component` | `int`，必填 | 成分编号 |
| **返回** | `ComponentProfile` | 可按字典使用的画像对象，支持 notebook 显示和 `to_html()` |

从本地或 Hugging Face 载入产物时，画像文件会按需延迟加载。

返回的成分画像仍支持普通字典索引，并包含上文所述的 `tail_direction`、
`dominant_sign`、`sign_statistics`、`score_statistics`、`examples`、`logit_lens` 和可选的
`r_lens`。如果拟合层记录了 logcosh 成分目标，返回的显示对象还会包含
`fitting_statistics`：目标值、高斯基线、绝对偏差和从 1 开始的偏差排名。该字段在读取时
从层拟合元数据派生，不会写入压缩的成分画像 JSON。在 notebook 中，把 `component` 放在
单元格末尾即可显示画像面板；也可调用
`component.to_html("component-188.html")` 保存。四项符号统计的含义如下：

| 字段 | 含义 |
| --- | --- |
| `positive_fraction` | 非零画像位置中分数为正的比例 |
| `negative_fraction` | 非零画像位置中分数为负的比例 |
| `positive_energy_fraction` | 该成分在整个画像语料上的分数平方总量中位于正侧的比例 |
| `negative_energy_fraction` | 该成分在整个画像语料上的分数平方总量中位于负侧的比例 |

### `top_occurrence(...)`

无需遍历完整的成分画像字典，即可读取一个已存储的高激活样例：

```python
occurrence = lens.top_occurrence(layer=7, component=452, id=0)
print(occurrence["context"], occurrence["text"], occurrence["score"])
```

样例 ID 是已存储排名中的零基位置。默认使用成分所选的尾部方向；也可以传入
`sign="positive"` 或 `sign="negative"` 明确选择一侧。

普通结果包含产物中保存的短上下文窗口。设置 `full=True` 后，方法会根据已记录的
数据集仓库、revision、split、文本字段和 `source_index` 读取完整的原始数据样本：

```python
occurrence = lens.top_occurrence(layer=7, component=452, id=0, full=True)
print(occurrence["full_text"])
```

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `layer` | `int`，必填 | 已建立画像的层 |
| `component` | `int`，必填 | 成分编号 |
| `id` | `int = 0` | 零基的已存储样例 ID |
| `sign` | `"positive" \| "negative" \| None = None` | 指定分数方向；`None` 使用所选尾部 |
| `full` | `bool = False` | 以 `full_text` 字段附加完整原始样本 |
| **返回** | `dict` | 已存储样例记录的独立副本 |

### `erf.suffix_sweep(...)`

测量一个已存储成分样例需要多少左侧上下文，才能恢复其所选尾部方向上的绝对分数排名：

```python
result = lens.erf.suffix_sweep(
    layer=6,
    component=214,
    rank_thresholds=(1, 3, 5, 10, 15),
    occurrences=20,
)
```

该方法使用成分画像中已存储的所选尾部样例，按需加载并复用 Lens 的语言模型与
tokenizer。返回结果包含每个排名阈值的成分汇总，以及逐样例的后缀扫描记录。该方法
不会写入文件。完成后可调用 `lens.unload_model()` 释放缓存的语言模型。

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `layer` | `int`，必填 | 已建立画像的 Transformer block 层 |
| `component` | `int`，必填 | 已存储的成分编号 |
| `rank_thresholds` | `Sequence[int] = (1, 3, 5, 10, 15)` | 一次扫描中记录的绝对分数排名阈值 |
| `occurrences` | `int = 20` | 最多测量的所选尾部样例数 |
| `exact_suffix_length` | `int = 10` | 逐一测试到该长度为止的所有后缀 |
| `max_batch_size` | `int = 64` | 最大样例批量大小 |
| `batch_token_budget` | `int = 64` | 每个 forward 批次的大致词法 token 预算 |
| `device` | `str \| torch.device \| None = "auto"` | 模型设备 |
| `model` | `torch.nn.Module \| None = None` | 可选的调用方语言模型 |
| `tokenizer` | tokenizer 或 `None` | 可选的调用方 tokenizer |
| `verbose` | `bool = False` | 是否报告模型加载和复用信息 |
| **返回** | `dict` | 阈值汇总和逐样例扫描记录 |

### `checkpoint_component_profile(...)`

立即保存某个已经在内存中完成的逐层画像：

```python
path = lens.checkpoint_component_profile(
    "icalens-output/my-icalens",
    layer=5,
)
```

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `path` | `str \| Path`，必填 | 已存在的本地 ICA Lens 产物目录 |
| `layer` | `int`，必填 | 要写入新画像的层 |
| **返回** | `Path` | 解析后的产物目录 |

该层必须已经通过 `profile_components()` 创建，或通过 `add_r_lens_profile()` 丰富了
内存画像。此方法会将压缩画像写入
`component_profiles/` 并更新 manifest，而不会重写无关层的张量。`icalens profile`
CLI 使用它逐层保存 checkpoint。普通的一次性 Python 流程调用 `save()` 即可。

## 变换激活张量

### `transform(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `activations` | 浮点 Tensor 或 ndarray，必填 | 最后一维等于 `lens.hidden_size` 的激活 |
| `layer` | `int`，必填 | 使用的已拟合层 |
| **返回** | 同输入数组类型 | 保留前导维度的带符号 ICA 分数 |

### `energy(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 带符号成分分数 |
| **返回** | 同输入数组类型 | 每个向量内的非负成分能量占比 |

### `keep_topk(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 成分分数 |
| `k` | `int`，必填 | 每个向量保留的最大绝对分数数量 |
| **返回** | 同输入数组类型 | 其余成分置零的副本 |

### `ablate_topk(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 成分分数 |
| `k` | `int`，必填 | 每个向量移除的最大绝对分数数量 |
| **返回** | 同输入数组类型 | 所选成分置零的副本 |

### `inverse_transform(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 最后一维等于成分数的带符号分数 |
| `layer` | `int`，必填 | 使用的 writing matrix 与中心 |
| **返回** | 同输入数组类型 | 重构的预处理激活 |

### `restore_norm(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `values` | 浮点 Tensor 或 ndarray，必填 | 重构或修改后的激活方向 |
| `reference` | 同类型、同形状，必填 | 用于提供每个向量目标范数的原始激活 |
| **返回** | 同输入数组类型 | 按 `reference` 恢复范数后的 `values` |

## 查看拟合过程

### `plot_fitting_curve(...)`

绘制拟合期间记录的、跨 ICA 成分的目标函数分布：

```python
figure = lens.plot_fitting_curve(layer=6)
figure  # 在 Jupyter 中直接显示

# 在同一张图中绘制多个独立的层面板
figure = lens.plot_fitting_curve(layers=[0, 6, 11], columns=3)
figure = lens.plot_fitting_curve(layers="all", columns=4)
```

```python
lens.plot_fitting_curve(
    *,
    layer=None,
    layers=None,
    columns=None,
) -> matplotlib.figure.Figure
```

| 参数 | 类型／默认值 | 含义 |
| --- | --- | --- |
| `layer` | `int \| None = None` | 绘制一个已拟合层；不能与 `layers` 同时使用 |
| `layers` | `list[int] \| tuple[int, ...] \| "all" \| None` | 绘制多个已拟合层；不能与 `layer` 同时使用 |
| `columns` | `int \| None = None` | 子图列数；默认为 2，且不会超过所选层数 |
| **返回值** | Matplotlib `Figure` | 分量百分位嵌套区间和目标函数中位数曲线 |

必须且只能传入 `layer` 或 `layers` 其中一个。选择多层时，每层使用独立子图，不会
聚合在一起。每个面板展示每个记录迭代中跨成分的最小值、第 10 至第 90 百分位数、
最大值和中位数。

该图完全由产物元数据生成，不会加载语言模型、拟合激活或层张量。所选层必须包含
`objective_history`；当前拟合命令默认记录该信息，记录间隔由 `--objective-every`
或 `objective_every` 控制。

## 拟合激活张量

### `fit(...)`

```python
lens.fit(
    activations,
    layer=6,
    n_components=None,
    max_iter=20,
    random_state=0,
    progress=True,
    device="cuda",
    batch_size=8192,
    objective_every=1,
    provenance={
        "dataset": {"repo_id": "owner/dataset", "split": "train"},
        "fitting_tokens": int(activations.shape[0]),
    },
)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `activations` | 浮点 Tensor 或 ndarray，必填 | 前导维度会展平为样本 |
| `layer` | `int`，必填 | 激活来源层的标签；不会自动捕获 |
| `n_components` | `int \| None = None` | 成分数；默认等于激活宽度 |
| `algorithm` | `str = "parallel"` | FastICA 更新算法 |
| `fun` | `str = "logcosh"` | 对比函数 |
| `max_iter` | `int = 200` | 固定迭代次数 |
| `random_state` | `int \| None = 0` | 初始化随机种子 |
| `progress` | `bool = False` | 显示进度 |
| `device` | device 或 `None` | 拟合设备 |
| `batch_size` | `int = 8192` | 每次在拟合设备处理的行数 |
| `objective_every` | `int = 1` | 每 N 次迭代记录目标分布 |
| `provenance` | `dict \| None = None` | 原样存入产物的 JSON 格式来源信息 |
| **返回** | `ICALens` | 拟合或替换该层后的同一个 Lens |

完整宽度的 ICA 至少需要 `hidden_size + 1` 个样本，因为中心化会使数据秩减少 1。

再次调用 `fit()` 会在同一个 Lens 中添加或替换一层。端到端数据集命令和完整 Python
示例见[拟合与发布](fit-and-publish.md)。

## 保存与发布

### `save(...)`

```python
path = lens.save("icalens-output/my-icalens")
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `path` | `str \| Path`，必填 | ICA Lens 产物的目标目录 |
| **返回** | `Path` | 解析后的目标目录 |

本地输出路径不会写入产物；`provenance` 中的内容会原样保存。

### `push_to_hub(...)`

```python
url = lens.push_to_hub(
    "owner/my-icalens",
    private=False,
    revision="main",
)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `repo_id` | `str`，必填 | `owner/name` 形式的目标仓库 |
| `private` | `bool \| None = None` | 新建仓库时是否设为私有 |
| `token` | `str \| bool \| None = None` | Hugging Face 身份认证 |
| `revision` | `str = "main"` | 目标 branch 或 revision |
| `commit_message` | `str` | Hub commit 消息 |
| **返回** | `str` | Hub commit 的 URL |

## 结果对象

### `CaptureResult`

`CaptureResult` 包含 `tokens`、`token_texts`、`token_labels`、`token_tooltips`、
`token_groups`、`token_ids`、`positions` 和 `activations`。

### `AnalysisResult`

`AnalysisResult` 还包含 `scores`、`energy`、`model`、`layer`、`input_text`、
`token_scope`、`messages`、`component_profiles` 和 `logit_effects`。其中
`component_profiles` 是当前分析层的精简逐成分画像；该层没有画像时为 `None`。
`logit_effects` 保存由 `lens.add_logit_effects(...)` 添加的逐 token 局部 Logit Lens
效应。

在 Jupyter 或 Colab 中，将 `result` 作为单元格最后一个表达式即可显示结果：

```python
result
```

`component_profiles` 为交互式结果提供符号统计、高能量样例、Logit Lens token 和可选
R-lens token。完整画像仍可通过 `lens.component_profile(...)` 读取。

### `add_logit_effects(...)`

分析输入后，可以将每个 token 上绝对值最大的若干成分分别放大，并把最大的直接
Logit Lens 变化附加到结果中：

```python
result = lens.analyze("She deposited the check at the bank.", layer=6)
result = lens.add_logit_effects(
    result,
    components_per_token=3,
    multiplier=1.1,
    effect_tokens_per_component=10,
)
result
```

这些数值显示在单独的 **Local intervention projection** 面板中。它们是经过最终
归一化层与 unembedding 后的直接投影，不会运行后续 Transformer 层。

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `result` | `AnalysisResult`，必填 | 已有分析结果；复用其中的激活与缓存模型 |
| `components_per_token` | `int = 3` | 每个已分析 token 上按分数绝对值选取的成分数 |
| `multiplier` | `float = 1.1` | 原始有符号分数的乘数 |
| `effect_tokens_per_component` | `int = 10` | 每次局部成分编辑按 logit 变化绝对值保留的词表 token 数 |
| `batch_size` | `int = 32` | 同时处理的局部编辑数；调小可降低临时显存 |
| `vocabulary_batch_size` | `int = 16384` | 以 float32 同时投影的词表行数；调小可降低临时显存 |
| **返回** | `AnalysisResult` | 包含局部效应的新结果 |

### `AnalysisResult.display(...)`

```python
result.display(metric="energy", top_k=5, height=720)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `metric` | `"score" \| "energy" = "score"` | 成分条所显示的指标 |
| `top_k` | `int = 3` | 每张 token 卡显示的成分数 |
| `title` | `str = "ICA Lens Explorer"` | 嵌入报告标题 |
| `height` | `int = 720` | 嵌入区域的初始高度（像素） |
| **返回** | `None` | 通过 IPython 显示交互结果 |

### `AnalysisResult.to_html(...)`

```python
output = result.to_html(
    "analysis.html",
    metric="score",
    top_k=3,
    title="ICA Lens Explorer",
)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `output_file` | `str \| Path`，必填 | 目标 HTML 文件 |
| `metric` | `"score" \| "energy" = "score"` | 成分条显示的值 |
| `top_k` | `int = 3` | 每张 token 卡显示的成分数 |
| `title` | `str = "ICA Lens Explorer"` | 报告标题 |
| **返回** | `Path` | 独立 HTML 报告路径 |

## 查看 Lens 信息

常用属性包括 `model_id`、`model_revision`、`model_type`、`activation_site`、
`hidden_size`、`available_layers` 和 `metadata`。

```python
print(lens.model_id)
print(lens.model_revision)
print(lens.model_type)
print(lens.activation_site)
print(lens.hidden_size)
print(lens.available_layers)
print(lens.metadata)
```

`metadata` 返回可移植 manifest 的独立字典副本。

## `ActivationDataset`

打开由 `icalens capture text` 或 `icalens capture chat` 保存的可复用激活值：

```python
from icalens import ActivationDataset

captured = ActivationDataset("/mnt/external/icalens-activations/gpt2-pile10k-1m")
values = captured.layer(6)
```

| 成员 | 类型 | 含义 |
| --- | --- | --- |
| `path` | `Path` | 激活值数据目录的绝对路径 |
| `available_layers` | `tuple[int, ...]` | 已捕获的 Transformer 层 |
| `sample_count` | `int` | 每层对齐的 token 行数 |
| `hidden_size` | `int` | 每行激活值的宽度 |
| `dtype` | `torch.dtype` | 磁盘中的激活值类型 |
| `model` | `dict` | 记录的模型标识、版本与类型 |
| `provenance` | `dict` | 数据集、采样、文档边界和格式来源 |
| `layer(layer)` | `torch.Tensor` | 磁盘映射的 `[sample_count, hidden_size]` 张量 |

## 异常

```python
from icalens import ArtifactError, ICALensError, NotFittedError
```

- `ICALensError`：本包所有异常的基类。
- `ArtifactError`：ICA Lens 产物缺失、损坏或不兼容。
- `NotFittedError`：请求的层尚未拟合或加载。

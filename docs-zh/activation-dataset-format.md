# ActivationDataset 格式

`ActivationDataset` 是 `icalens fit activations` 使用的可移植、磁盘后端格式。外部推理
系统可以直接写出这种格式，无需使用 ICA Lens 加载模型或捕获激活值。

本文规定格式版本 1，其标识为：

```json
{
  "format": "icalens.activations",
  "format_version": 1
}
```

## 目录结构

```text
my-activations/
├── activations.json
├── samples.safetensors
└── layers/
    ├── layer_00.bfloat16.bin
    ├── layer_01.bfloat16.bin
    └── layer_02.bfloat16.bin
```

`layers/` 下的文件名由清单指定，不必遵循示例中的命名方式。清单中的路径均相对于
数据集目录。

## 激活值

请在从零开始编号的 Transformer block 上捕获 block 输出后的残差流（`resid_post`）。
每层写入一个连续、无文件头、按行优先排列的张量，形状为：

```text
[sample_count, hidden_size]
```

支持的元素类型为 `bfloat16`、`float16`、`float32` 和 `float64`。ICA Lens 自带的捕获
命令使用 `bfloat16`。原始文件必须严格包含 `sample_count * hidden_size` 个元素，并
采用清单声明的 dtype。

每个层文件的第 `i` 行，以及 `samples.safetensors` 中每个张量的第 `i` 行，都必须表示
同一个采样 token。采样顺序不限，但必须保持这种对齐关系。

`resid_post` 指模型最终归一化之前的 Transformer block 输出。不要把 embedding 输出
当作第 0 层：第 0 层是 Transformer block 0 的输出。

## `samples.safetensors`

样本元数据是一个 Safetensors 文件，必须包含以下三个长度均为 `sample_count` 的一维
`int64` 张量：

| 张量 | 含义 |
| --- | --- |
| `document_index` | 捕获数据流中源文档或渲染后对话的从零开始编号 |
| `position` | 该文档完整模型输入中的从零开始 token 位置 |
| `token_id` | 该位置对应的 tokenizer ID |

`position` 指实际传入模型的输入，包括原生聊天模板、BOS、EOS 或其他边界 token。
被排除在采样之外的上下文 token 仍保留在模型输入中，但没有激活值行，因此也不会在
该文件中出现。

允许添加其他对齐张量。除非另有文档说明，读取方不应依赖这些额外张量。

## `activations.json`

清单文件必须命名为 `activations.json`。下面是一个包含三层的完整示例：

```json
{
  "activation_site": "resid_post",
  "dtype": "bfloat16",
  "format": "icalens.activations",
  "format_version": 1,
  "hidden_size": 3072,
  "layer_indexing": "transformer_blocks_zero_based",
  "layers": {
    "0": {
      "file": "layers/layer_00.bfloat16.bin",
      "shape": [1000000, 3072],
      "status": "complete"
    },
    "23": {
      "file": "layers/layer_23.bfloat16.bin",
      "shape": [1000000, 3072],
      "status": "complete"
    },
    "47": {
      "file": "layers/layer_47.bfloat16.bin",
      "shape": [1000000, 3072],
      "status": "complete"
    }
  },
  "model": {
    "repo_id": "Qwen/Qwen3.5-122B-A10B",
    "revision": "FULL_COMMIT_HASH",
    "type": "instruct"
  },
  "provenance": {
    "dataset": {
      "description": "Neuronpedia fitting corpus"
    },
    "fitting_tokens": 1000000,
    "sampling_seed": 0,
    "token_scope": "all"
  },
  "sample_count": 1000000,
  "samples_file": "samples.safetensors",
  "status": "complete"
}
```

各字段要求如下：

| 字段 | 要求 |
| --- | --- |
| `format` | 必须为 `icalens.activations` |
| `format_version` | 必须为 `1` |
| `status` | 开始拟合前必须为 `complete` |
| `model.repo_id` | 写入最终 Lens 的模型标识符 |
| `model.revision` | 模型的精确 revision；建议使用完整 commit hash |
| `model.type` | `base` 或 `instruct` |
| `activation_site` | 标准 ICA Lens 流程必须使用 `resid_post` |
| `layer_indexing` | 使用 `transformer_blocks_zero_based` |
| `sample_count` | 每个捕获层中的激活值行数 |
| `hidden_size` | 每行激活值的元素数 |
| `dtype` | `bfloat16`、`float16`、`float32` 或 `float64` |
| `samples_file` | 样本元数据 Safetensors 文件的相对路径 |
| `layers` | 以十进制、从零开始的层号为键的对象 |
| `layers.*.file` | 该层原始二进制张量的相对路径 |
| `layers.*.shape` | `[sample_count, hidden_size]` |
| `layers.*.status` | 必须为 `complete` |
| `provenance` | 描述源数据、样本选择、token 边界格式和采样过程的 JSON 对象 |

`provenance` 对象可以扩展。请记录足够的信息以复现模型输入和采样行，包括数据集标识
与 revision、输入字段、聊天模板或文档边界格式、token scope、上下文长度、采样策略、
token 预算和随机种子等。

## 验证与拟合

开始长时间拟合前，先加载一次该目录：

```python
from icalens import ActivationDataset

dataset = ActivationDataset("/path/to/my-activations")
print(dataset.available_layers)
print(dataset.sample_count, dataset.hidden_size, dataset.dtype)
print(dataset.layer(dataset.available_layers[0]).shape)
print(dataset.samples().keys())
```

这会检查格式版本、完成状态、支持的 dtype、必需的样本元数据、元数据长度，并在打开
某一层时检查层文件大小。之后即可在不加载语言模型的情况下进行拟合：

```bash
icalens fit activations \
  --input /path/to/my-activations \
  --layers all \
  --icalens-preprocessing none \
  --max-iter 50 \
  --fit-batch-size 16384 \
  --output /path/to/icalens-model
```

拟合时每次只 memory-map 一个层。全宽 Lens 至少需要 `hidden_size + 1` 行激活值；正式
发布的 Lens 通常会使用远多于这一最低要求的样本。

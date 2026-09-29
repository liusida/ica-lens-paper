# 快速开始

本页将从安装开始，带你完成一次文本或对话的交互式分析。

## 安装

ICA Lens 需要 Python 3.10 或更高版本。运行待分析的语言模型时推荐使用 CUDA GPU；
也支持使用 CPU，但速度会慢一些。

```bash
pip install icalens
```

访问 Hugging Face 上的公开模型无需登录。访问私有或受限仓库时必须进行身份认证；
登录后还可以获得更高的下载速率限制。

可以在 [ICA Lens 模型合集](https://huggingface.co/collections/sida/ica-lens)中选择已经发布的
Lens。

## 分析文本

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze("She deposited the check at the bank.", layer=6)

result
```

在 Jupyter 或 Colab 中，将 `result` 放在单元格的最后一行，即可显示交互式的逐 token
分析结果。第一次分析时，程序会加载语言模型和指定的 ICA Lens 层。之后使用同一个
`lens` 继续分析时，会复用内存中已经加载的模型。

默认的 `device="auto"` 会在 CUDA 可用时使用 CUDA，否则使用 CPU。也可以显式指定
`device="cuda"` 或 `device="cpu"`。

## 查看结果

在 Jupyter 或 Colab 中，结果会显示为交互式的逐 token 分析界面。

![在 Jupyter 中进行 ICA Lens 逐 token 分析](https://icalens.readthedocs.io/en/latest/assets/text-analysis-notebook.png)

- 每张卡片表示模型指定层上的一个 token。
- 例如，`C37` 表示编号为 37 的 ICA 成分。
- **Score** 显示带符号的成分激活，因此保留了方向信息。
- **Energy** 显示该成分在当前 token 分数平方和中所占的比例。
- 点击一个成分，即可在所有显示的 token 上高亮该成分。

### 查看成分画像

如果所选成分带有画像，可以展开 token 卡片下方的 **Component profile**。

![包含语料样例、Logit Lens 和 R-lens 读出的 ICA 成分画像](assets/component-profile-r-lens.png)

- 画像标题显示由分数偏度选择的尾部。
- 紧凑统计栏显示 **Skewness**、**Excess kurtosis**，以及用于排列成分 ID 的拟合时
  **Logcosh deviation**。
- **Top-score occurrences** 给出所选尾部中成分绝对分数最高的代表性位置。结合高亮 token 及其上下文，
  可以推测成分可能表达的含义。
- **Logit-lens tokens** 显示该成分写入方向会提升的词表 token。这些 token 只是辅助
  线索，不能当作确定标签，也不是对模型生成结果的预测。
- **R-lens tokens**（如果存在）近似纳入后续 Transformer block 的平均线性影响，可作为
  另一类读出证据。

token 卡片上方的控件可以切换指标并调整显示的成分数量。阅读时可以先看高分样例，
检查不同上下文是否表达一致，再比较 Logit Lens 和 R-lens token 作为补充证据。

## 分析另一个输入

复用同一个 Lens 可以比较不同输入上的成分，而且无需再次加载语言模型：

```python
result = lens.analyze("She walked along the river bank.", layer=6)
result
```

不再需要缓存的模型和 tokenizer 时，可以调用 `lens.unload_model()` 释放它们占用的
内存。

## 分析对话

指令模型的 Lens 接受包含 `role` 和 `content` 字段的消息：

```python
from icalens import ICALens

lens = ICALens.from_pretrained(
    "sida/icalens-qwen3.5-2b-ultrachat-1m"
)

result = lens.analyze(
    [
        {"role": "user", "content": "What is the most interesting science?"},
        {"role": "assistant", "content": "Physics."},
    ],
    layer=16,
)

result
```

![在 Jupyter 中按消息分组显示 ICA Lens 分析结果](https://icalens.readthedocs.io/en/latest/assets/conversation-analysis-notebook.png)

程序会自动应用 tokenizer 的聊天模板。默认情况下，分析结果包括消息内容、角色标记、
分隔符以及其他模板 token。`analyze()` 只分析传入的对话，不会生成新的助手回复。

## 保存 ICA Lens Explorer

可以将结果保存为独立的 ICA Lens Explorer，供 notebook 之外的环境使用：

```python
result.to_html("analysis.html")
```

## 下一步

- 了解[文本与对话输入](text-and-chat.md)。
- 理解[分数与能量](scores-and-energy.md)。
- 通过[重构](reconstruction.md)将成分分数映射回激活空间。
- [拟合并发布](fit-and-publish.md)自己的 ICA Lens。

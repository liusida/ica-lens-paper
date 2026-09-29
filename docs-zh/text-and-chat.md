# 文本与对话

`lens.analyze()` 既可以接收纯文本，也可以接收已有回复的完整对话。输入形式决定了
ICA Lens 是否应用聊天模板，以及可以选择哪些 token 范围。

## 纯文本

传入字符串时，ICA Lens 会将它作为普通模型输入直接分析：

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze(
    "The boat reached the bank before sunset.",
    layer=6,
)

result
```

程序不会添加聊天模板、角色标记或对话分隔符。字符串编码后得到的每个 token 都会被
捕获和分析。

![ICA Lens 纯文本分析](https://icalens.readthedocs.io/en/latest/assets/text-analysis-notebook.png)

即使待分析的是指令模型，传入字符串也会绕过它的聊天模板。如果希望采用模型进行对话
时使用的格式，请传入消息列表。

## 对话

对话应以消息列表的形式传入，每条消息都包含 `role` 和 `content` 字段：

```python
from icalens import ICALens

lens = ICALens.from_pretrained(
    "sida/icalens-qwen3.5-2b-ultrachat-1m"
)

messages = [
    {"role": "user", "content": "What is the most interesting science?"},
    {"role": "assistant", "content": "Physics."},
]

result = lens.analyze(messages, layer=16)
result
```

支持的角色包括 `system`、`user` 和 `assistant`。ICA Lens 会先检查消息格式，再使用待
分析模型 tokenizer 中的聊天模板渲染这些消息。最终的 token 序列可能包含：

- 角色标记和消息边界标记；
- 换行符和分隔符；
- user、system 和 assistant 的消息内容；
- 思考标记等模型特有的控制 token。

在 notebook 中，结果会按照消息分组；与每条消息相关的模板 token 也会保留在对应
分组中。

![按消息分组的 ICA Lens 分析](https://icalens.readthedocs.io/en/latest/assets/conversation-analysis-notebook.png)

## Token 范围

对话分析默认使用 `token_scope="all"`。如果研究问题不涉及模板 token 或其他角色的
token，可以选择更窄的范围。

| Scope | 包含的 token |
| --- | --- |
| `"all"` | 渲染后的完整对话，包括模板 token |
| `"content"` | 所有角色的消息内容 |
| `"user"` | 仅用户消息内容 |
| `"assistant"` | 仅助手消息内容 |

例如：

```python
assistant_result = lens.analyze(
    messages,
    layer=16,
    token_scope="assistant",
)
```

Token 范围只会改变结果中返回的位置，不会改变送入模型的对话。计算每个被选中位置的
激活时，模型仍然使用渲染后的完整对话作为上下文。

模板边界由 tokenizer 决定。ICA Lens 通过 fast tokenizer 提供的字符偏移量对齐消息
内容，因此对话分析要求 tokenizer 同时支持 fast 模式和聊天模板。如果所选范围中没有
任何 token，`analyze()` 会报错，而不是返回空结果。

## 多轮对话

按时间顺序传入完整的对话历史：

```python
messages = [
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "Name an interesting science."},
    {"role": "assistant", "content": "Physics."},
    {"role": "user", "content": "I prefer biology."},
    {"role": "assistant", "content": "Then consider genetics."},
]

result = lens.analyze(messages, layer=16)
```

结果会分别显示 `System`、`User 1`、`Assistant 1`、`User 2` 和 `Assistant 2` 等 token
分组。每个激活都包含上下文信息：计算后续轮次中某个 token 的激活时，模型已经处理了
之前的全部对话。

## 分析不会生成回复

`analyze()` 只捕获已有内容的激活，不会要求语言模型继续生成新的助手回复。

如果要分析生成的文本：

1. 使用你选择的推理接口生成回复。
2. 将回复以 `assistant` 角色追加到 `messages`。
3. 把完整对话传给 `lens.analyze()`。

这样可以让温度、最大生成 token 数等生成设置与激活分析相互独立。

## 上下文长度与截断

使用 `context_length` 可以在模型前向传播前截断编码后的输入：

```python
result = lens.analyze(
    messages,
    layer=16,
    context_length=2048,
)
```

对于对话输入，程序会先渲染聊天模板，再进行截断。返回的 `result.positions` 表示 token
在渲染后序列中的位置。如果某条消息的内容完全落在保留的上下文之外，它就不会在结果
中贡献任何 token。

## 检查 token 对齐

结果为每个被选中的位置保留了多种相互对齐的表示：

```python
result.tokens       # tokenizer vocabulary forms
result.token_texts  # individually decoded text
result.token_ids    # integer token IDs
result.positions    # positions in the model input sequence
result.scores       # signed ICA scores, one row per token
result.energy       # per-token component energy shares
```

这些字段的第一个维度都是相同的 token 维度。部分多语言 tokenizer 会把一个可见字符
拆成多个字节片段。对于无法单独解码的片段，交互界面会显示 `<?>`；将鼠标悬停在该
token 上，可以看到 token ID，以及把相邻片段合并解码后得到的可读文本。

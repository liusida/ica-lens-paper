# 引导

在把 ICA 坐标映射回残差流之前，可以先修改这些坐标。本页复现一个定性干预：让
Qwen 3.5 2B 的回答从 **Quantum Computing** 变为 **Neuroplasticity**。

## 运行示例

请在 Jupyter 或 Colab 中按顺序运行以下单元格。

### 1. 生成基线回答

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-qwen3.5-2b-ultrachat-1m")
messages = [{
    "role": "user",
    "content": "If you had to pick one, what is the most interesting science? Be brief.",
}]

baseline = lens.generate(messages, max_new_tokens=16)
print("Baseline:", baseline)
```

在我们测试的模型 revision 上，贪心解码的回答以 **Quantum Computing** 开头。

### 2. 检查成分及其所选尾部

首先读取由画像范围内分数偏度选择的尾部：

```python
component = lens.component_profile(layer=5, component=188)
print(component["tail_direction"])
print(component["sign_statistics"])
print(component["examples"][component["tail_direction"]]["tokens"][:10])
```

在这个产物中，`C188` 所选的尾部是 `negative`。画像根据总体偏度选择这一侧，并给出
该侧的代表性样例；但它本身不能证明某个具体概念属于这一方向，因此
还需要用相互独立的探针验证解释。

请用相互独立的输入分析不同概念，不要把它们写进同一个列表。对于因果语言模型，前面
出现的概念会影响后续所有 token。

```python
neuroscience_result = lens.analyze("Neuroscience.", layer=5)
neuroscience_result
```

在另一个单元格中运行：

```python
quantum_result = lens.analyze("Quantum computing.", layer=5)
quantum_result
```

对于这个特定 Lens 的第 5 层，“Neuroscience” 中的 `uro` 和 `science` token 上，
`C188` 显著为负（约为 `-8.7` 和 `-14.9`）；而在“Quantum computing.”中接近零。
“The human brain.”和“Neuroplasticity.”等独立探针也显示相同的负方向。

这里的负号同时得到已保存画像和独立探针的支持，而不是从“Neuroscience”这个标签
推断出来的。ICA 的符号是任意的；另一个独立拟合的 Lens 可能用相反的符号约定表示
同一方向。因此，应始终读取当前所用 Lens 自己的画像。

### 3. 检查基线对话

把生成的回答追加到消息中，再分解完整对话：

```python
baseline_messages = [
    *messages,
    {"role": "assistant", "content": baseline},
]
baseline_result = lens.analyze(baseline_messages, layer=5)
baseline_result
```

### 4. 钳制并重新生成

在每个被处理的 token 位置和每个自回归步骤上，把 `C188` 钳制到 `-20`：

```python
steered = lens.generate(
    messages,
    layer=5,
    clamp=(188, -20.0),
    max_new_tokens=16,
)
print("Steered:", steered)
```

在我们的测试中，回答以 **Neuroplasticity** 开头。多次调用会复用内存中的语言模型
权重。

### 改为增加分数偏移

钳制会设置一个绝对坐标值；加性引导则把坐标移动一个经过校准的偏移量：

```python
steered = lens.generate(
    messages,
    layer=5,
    steer=(188, -12.0),
    steering_scope="current-position",
    max_new_tokens=16,
)
```

这里，`steer=(188, -12.0)` 执行

$$
h \leftarrow h - 12 A_{:,188},
$$

其中 $A_{:,188}$ 是该成分的写入方向。默认的 `current-position` 会在预填充时修改
提示词的最后一个位置，并在每个解码步骤修改新生成位置，通常是最经济的干预方式。
使用 `steering_scope="all-positions"` 时，同一偏移还会在预填充时加到提示词的每个
位置。

加性引导要求 Lens 在拟合时没有逐行归一化，也没有预处理中心。`clamp` 与 `steer`
不能同时使用。

### 5. 检查生成后的对话

```python
steered_messages = [
    *messages,
    {"role": "assistant", "content": steered},
]
steered_result = lens.analyze(steered_messages, layer=5)
steered_result
```

最后这次调用是对干预所生成文本进行的普通、无引导分析。它展示原始模型如何自然表示
这段对话，并不会重放或追踪生成时使用的钳制。

生成默认使用贪心解码。模型、Lens、Transformers 版本、硬件、提示词或生成参数变化
时，具体续写也可能改变。

## 校准目标值

ICA 分数是一个坐标，不是语义剂量。绝对值更大并不保证概念效果更强或单调变化。相比
随意选择一个数值，进行短距离扫描更有信息量：

```python
for target in (-5.0, -10.0, -15.0, -17.0, -20.0, -25.0):
    response = lens.generate(
        messages,
        layer=5,
        clamp=(188, target),
        max_new_tokens=16,
    )
    print(target, response.splitlines()[0])
```

在我们的运行中，回答从 Quantum Computing 逐渐经过 Quantum Biology 和 The Human
Brain，最后变为 Neuroplasticity。这个扫描只是针对一个产物和提示词的经验校准，并非
通用刻度。

## 钳制如何工作

在选定的残差流层上，ICA Lens 会在每个被处理的 token 位置和每个自回归步骤执行：

1. 按照 ICA 拟合时的方式，对隐藏状态进行 L2 归一化。
2. 读取带符号的 ICA 分数。
3. 将选定坐标替换为目标值。
4. 在分数空间中保持其他坐标不变。
5. 通过写入矩阵映射修改后的分数。
6. 恢复原残差向量的范数，再把修改后的向量送回模型。

参数 `clamp=(188, -20.0)` 表示请求

\[
s_{188} \leftarrow -20.
\]

恢复范数会重新缩放重构向量，因此再次变换最终激活时，所得分数可能接近、但不完全
等于 `-20`。

对于已有的隐藏状态张量，等价修改为：

```python
scores = lens.transform(hidden_states, layer=5)
edited_scores = scores.clone()
edited_scores[..., 188] = -20.0
normalized_edit = lens.inverse_transform(edited_scores, layer=5)
edited_hidden_states = lens.restore_norm(
    normalized_edit,
    reference=hidden_states,
)
```

无论使用钳制还是加性引导，`lens.generate()` 都会自动安装并移除模型 hook。

## 实践注意事项

- 引导时使用带符号的分数，不要使用能量占比。
- 成分标签只是由示例支持的假设，不是内置含义或类别标签。
- 先读取已保存的尾部方向，再用相互独立的探针确认具体概念的方向；绝不能根据标签
  猜测符号。
- 模型 revision、层、激活位置和预处理必须与拟合 Lens 中记录的信息一致。
- 校准适中的目标值，并与确定性的基线比较。
- 很大的目标值可能显著旋转残差向量，引发非线性或退化行为。
- 一次定性变化并不表示某个成分能在所有上下文中确定性地控制一个概念。

逆变换与范数恢复详见[重构](reconstruction.md)。

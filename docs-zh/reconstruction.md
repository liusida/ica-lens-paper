# 重构

ICA Lens 可以把成分分数映射回拟合时使用的激活坐标：

```python
normalized_reconstruction = lens.inverse_transform(result.scores, layer=6)
```

!!! note
    `inverse_transform()` 必须使用带符号的分数（`result.scores`），不能使用能量占比
    （`result.energy`）。

设 \(A\) 为拟合得到的写入矩阵，\(\mu\) 为拟合中心，则逆变换为：

\[
\hat{x} = sA^\top + \mu
\]

返回张量的前导维度和激活宽度，与生成这些分数的输入相同。

## 重构的是什么？

按照 ICA Lens 论文，默认预处理会在每个 token 位置上独立进行 L2 归一化：

\[
\tilde{x} = \frac{x}{\lVert x \rVert_2}
\]

归一化后的激活先被中心化和白化，再旋转到 ICA 坐标。白化变换与 ICA 旋转已经合并在
保存的读取矩阵中；`inverse_transform()` 使用的写入矩阵则通过这个组合变换映射回去。

因此，`inverse_transform()` 首先重构的是 \(\tilde{x}\)，即归一化后的预处理激活。
L2 归一化会丢失标量 \(\lVert x\rVert_2\)，所以仅凭分数无法恢复原始激活的范数。

```python
captured = lens.capture("She deposited the check.", layer=6)
scores = lens.transform(captured.activations, layer=6)
normalized_reconstruction = lens.inverse_transform(scores, layer=6)
```

对于满秩 Lens，除数值误差外，重构结果应当接近预处理后的激活。如果 ICA 成分数少于
激活宽度，重构就是低维近似，必然会丢失一部分信息。

## 恢复隐藏状态的范数

拟合好的 Lens 无法预先知道未来输入的激活范数，但 `analyze()` 会把原始捕获的隐藏
状态保存在 `result.activations` 中。因此，可以在变换前记录这些范数，并在重构后恢复：

```python
normalized_reconstruction = lens.inverse_transform(result.scores, layer=6)
restored_hidden_states = lens.restore_norm(
    normalized_reconstruction,
    reference=result.activations,
)
```

对于满秩 Lens，除拟合误差和浮点误差外，`restored_hidden_states` 应当能够很好地复现
捕获的隐藏状态。这个缩放依赖于具体输入：归一化方法保存在 Lens 中，而范数数值来自
当前正在分析的隐藏状态。

## 修改一个成分

可以先复制并修改带符号的分数，再进行重构。能量可用于选择显著成分，但实际修改和
逆变换必须作用于 `result.scores`：

```python
modified_scores = result.scores.clone()
modified_scores[:, 37] = 0

modified_normalized_reconstruction = lens.inverse_transform(modified_scores, layer=6)
modified_hidden_states = lens.restore_norm(
    modified_normalized_reconstruction,
    reference=result.activations,
)
```

这段代码先在 ICA 分数空间中移除成分 37，再恢复每个 token 原有的隐藏状态范数。也可以
用类似方法增大、减小或替换某个成分的分数。

使用原始范数可以让干预主要改变方向，避免意外改变残差流的整体幅度。对于可实际应用
的隐藏状态修改，这是一个合理的默认选择，但它仍然是一项干预设计：如果分数改动很大，
恢复原始范数可能会对改动效果进行一定程度的重新缩放。

重构后的向量不会被自动写回语言模型。要做到这一点，需要另行实现干预机制，并确保
模型、层、激活位置和 token 位置都与 Lens 一致。对于幅度很大或分布外的分数修改，应当
谨慎处理：逆变换是线性的，但模型后续的行为并不是线性的。

## 保留或消融 Top-K 成分

ICA Lens 提供逐 token 的 Top-K 分数掩码。成分按照分数绝对值排序，保留下来的带符号
数值不会改变。

在每个 token 位置只保留最强的十个成分：

```python
top_scores = lens.keep_topk(result.scores, k=10)
top_normalized_reconstruction = lens.inverse_transform(top_scores, layer=6)
top_hidden_states = lens.restore_norm(
    top_normalized_reconstruction,
    reference=result.activations,
)
```

消融最强的十个成分，并使用其余成分进行重构：

```python
remaining_scores = lens.ablate_topk(result.scores, k=10)
ablated_normalized_reconstruction = lens.inverse_transform(remaining_scores, layer=6)
ablated_hidden_states = lens.restore_norm(
    ablated_normalized_reconstruction,
    reference=result.activations,
)
```

每个 token 位置都会独立选择自己的 Top-K 成分。这两个方法都会返回新数组，不会修改
`result.scores`，并且保留输入的形状、dtype、设备以及 NumPy 或 PyTorch 类型。

## 衡量重构质量

### 余弦相似度

如果拟合 Lens 时采用逐行归一化，余弦相似度是很有用的指标：

```python
import torch.nn.functional as F

top_scores = lens.keep_topk(result.scores, k=10)
top_normalized_reconstruction = lens.inverse_transform(top_scores, layer=6)

similarity = F.cosine_similarity(
    F.normalize(result.activations, dim=-1),
    top_normalized_reconstruction,
    dim=-1,
)
```

取值范围为 `-1` 到 `1`；`1` 表示重构激活与原始激活的方向完全相同。每个 token 都会
得到一个相似度。

### 归一化均方误差（Normalized MSE）

对于每个 token，用它的重构平方误差除以“预测当前结果中归一化激活均值”所产生的
平方误差。这里应使用 `inverse_transform()` 输出的归一化结果，而不是
`restore_norm()` 返回的隐藏状态。

对于 token \(i\)，定义

\[
z_i = \frac{x_i}{\lVert x_i \rVert_2},
\qquad
\mu = \frac{1}{n}\sum_i z_i,
\]

并令 \(\hat{z}_i\) 为它在归一化空间中的重构。逐 token 的 Normalized MSE 为：

\[
\operatorname{NMSE}_i =
\frac{\lVert z_i - \hat{z}_i \rVert_2^2}
     {\lVert z_i - \mu \rVert_2^2}.
\]

```python
import torch.nn.functional as F

top_scores = lens.keep_topk(result.scores, k=10)
top_normalized_reconstruction = lens.inverse_transform(top_scores, layer=6)

normalized = F.normalize(result.activations, dim=-1)
baseline = normalized.mean(dim=0, keepdim=True)

token_squared_error = (
    normalized - top_normalized_reconstruction
).square().sum(dim=-1)
token_baseline_error = (normalized - baseline).square().sum(dim=-1)
token_normalized_mse = token_squared_error / token_baseline_error.clamp_min(1e-12)

print(token_normalized_mse)  # one value per token
```

解释如下：

- `0` 表示精确重构；
- `1` 表示重构平方误差，与使用当前结果中所有 token 的归一化激活均值来替换该 token
  时的平方误差相同；
- 小于 `1` 表示优于该基线；
- 大于 `1` 表示差于该基线。

论文中汇总报告的 Normalized MSE 是“求和后的比值”，而不是逐 token 比值的总和或
平均值：

\[
\operatorname{NMSE} =
\frac{\sum_i \lVert z_i - \hat{z}_i \rVert_2^2}
     {\sum_i \lVert z_i - \mu \rVert_2^2}.
\]

```python
normalized_mse = (
    token_squared_error.sum()
    / token_baseline_error.sum().clamp_min(1e-12)
)
```

当某个 token 本身非常接近均值时，基线误差接近零，它的单独比值可能会很大。对于一段
文本、一个模型或某一层的整体结果，汇总比值通常更稳定。以上示例假设 Lens 使用默认的
逐行归一化；如果设置了 `row_normalize=False`，应改用 `result.activations` 作为目标。

## 为什么不能从能量重构？

能量占比丢失了每个成分的符号，也丢失了分数向量的整体幅度。因此，许多不同的分数
向量会得到相同的能量占比，无法从 `result.energy` 唯一地还原激活。重构必须从
`result.scores` 等带符号的 ICA 分数开始。

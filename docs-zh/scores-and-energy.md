# 分数与能量

对于每个被分析的 token，ICA Lens 提供两种查看同一组成分坐标的方式：带符号的 ICA
分数，以及归一化后的能量占比。

## ICA 分数

`result.scores` 保存标准的带符号 ICA 坐标。设某个 token 的激活向量为 \(x\)，ICA Lens
会先应用拟合时记录的预处理，再将其投影到 ICA 成分上：

\[
s = W(\tilde{x} - \mu)
\]

按照 ICA Lens 论文，默认预处理是对每个 token 进行 L2 归一化：

\[
\tilde{x} = \frac{x}{\lVert x \rVert_2}
\]

其中，\(\mu\) 是归一化激活的拟合均值，\(W\) 是读取矩阵，\(s_j\) 是第 \(j\) 个成分
的分数。

### 白化在哪里进行？

FastICA 在白化后的坐标中拟合。设 \(K\) 是根据中心化并经过 L2 归一化的拟合激活计算
得到的白化矩阵，\(U\) 是在白化空间中学到的 ICA 旋转（解混矩阵）。ICA Lens 将二者
的乘积保存为读取矩阵：

\[
W = UK
\]

因此，完整的分数变换为：

\[
s = UK(\tilde{x} - \mu) = W(\tilde{x} - \mu)
\]

所以，虽然分析时没有单独调用白化操作，但白化已经包含在 `lens.transform()` 中。
拟合产物保存的是组合后的读取矩阵，用户无需手动依次应用两个矩阵。

```python
scores = result.scores
print(scores.shape)  # [tokens, components]
```

绝对值较大的正分数或负分数表示该 token 在对应成分方向上的投影较强。以下情况适合使用
分数：

- 同时关注激活方向和幅度；
- 比较正激活与负激活；
- 研究不同 token 上的成分分布；
- 重构或修改成分。

ICA 成分的符号并不唯一：将某个成分及其全部分数同时取反，仍然是同样有效的 ICA
解。拟合产物保存后，其符号会保持稳定，因此同一个 Lens 内部的比较仍然有意义。对于
独立拟合的不同 Lens，不应假设它们的成分编号、符号或原始分数量纲彼此对齐。

## 能量占比

对于成分分数 \(s_j\)，ICA Lens 将它在某个 token 上的能量占比定义为：

\[
e_j = \frac{s_j^2}{\sum_k s_k^2}
\]

```python
energy = result.energy
print(energy.shape)          # [tokens, components]
print(energy.sum(dim=-1))    # approximately 1 for each nonzero row
```

平方值衡量每个坐标对分数向量欧氏长度平方的贡献。再除以总和，就得到非负的相对占比；
对于每个非零 token 位置，这些占比之和为 1。

能量值为 `0.20`，即 20%，表示：

> 在这个 token 位置上，该成分贡献了 ICA 分数平方总量的 20%。

当正负分数可能相互抵消时，能量有助于判断哪些成分主导了当前 token。它不是概率、
置信度，也不是语义重要性的估计，而且不保留符号信息。

如果分数向量全部为零，ICA Lens 会返回全零的能量向量，而不会进行除零运算。

## 使用分数还是能量？

| 问题 | 使用 |
| --- | --- |
| 成分是正向激活还是负向激活？ | 分数 |
| 一个带符号的成分激活有多大？ | 分数 |
| 哪些成分主导了这个 token 的分数向量？ | 能量 |
| 某个成分占分数平方总量的多少？ | 能量 |
| 在整个画像语料中，成分的不对称尾部在哪一侧？ | 画像偏度 |
| 是否需要重构或修改激活？ | 分数 |

交互式结果默认显示分数。如果更关心同一个 token 内各成分的相对集中程度，可以选择
**Energy**。

## 在 Python 中查看数值

找出每个 token 上绝对值最大的带符号成分：

```python
values, components = result.scores.abs().topk(3, dim=-1)
```

找出能量占比最大的成分：

```python
shares, components = result.energy.topk(3, dim=-1)
```

两组数组的第一个维度都与 `result.tokens` 对齐，因此第 `i` 行始终对应
`result.tokens[i]`。

如果需要将分数映射回激活空间，请继续阅读[重构](reconstruction.md)。

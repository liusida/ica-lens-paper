# ICA Lens

ICA Lens 可以高效地将语言模型激活分解为相互独立的成分。与训练 SAE 字典相比，
拟合 ICA Lens 所需的计算量要小得多。它提供简洁的 Python API，用于加载、拟合、
共享和应用面向基础模型与指令模型的 ICA 变换。

```bash
pip install icalens
```

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze("She deposited the check at the bank.", layer=6)

result
```

可以在 [ICA Lens 模型合集](https://huggingface.co/collections/sida/ica-lens)中查看其他
已发布的 Lens。

![在 Jupyter 中进行 ICA Lens 逐 token 分析](https://icalens.readthedocs.io/en/latest/assets/text-analysis-notebook.png)

*在 Jupyter 中交互式查看每个 token 的成分。*

在 Jupyter 或 Colab 中，将 `result` 放在单元格最后一行即可直接显示交互界面。
使用 `result.to_html("analysis.html")` 可以保存独立的 HTML 文件。

[开始使用](getting-started.md){ .md-button .md-button--primary }
[阅读论文](https://arxiv.org/abs/2606.11722){ .md-button }

## ICA Lens 提供什么

- 为基础模型或指令模型的激活拟合 ICA Lens。
- 在任意已拟合层分析纯文本和多轮对话。
- 交互式查看带符号的成分分数和逐 token 能量占比。
- 用高能量样例、Logit Lens token 和可选的 R-lens 读出辅助标注成分。
- 通过本地目录或 Hugging Face Hub 保存、加载和分享 Lens。
- 在控制 GPU 和 CPU 内存占用的同时，扩展到大型 token 集合。

## 作者

- [Sida Liu](https://liusida.com/)
- [Feijiang Han](https://feijianghan.com/)

## 引用

```bibtex
@article{liu2026icalens,
  title={ICA Lens: Interpreting Language Models Without Training Another Dictionary},
  author={Liu, Sida and Han, Feijiang},
  journal={arXiv preprint arXiv:2606.11722},
  year={2026}
}
```

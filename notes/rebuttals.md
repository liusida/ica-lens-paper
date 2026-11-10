# ICA Lens Rebuttals



## **Official Review of Submission3245 by Reviewer DSqt**

Official Reviewby Reviewer DSqt19 Oct 2026, 14:13 (modified: 06 Nov 2026, 07:15)Everyone[Revisions](https://openreview.net/revisions?id=wdkt2YAUyD)

**Paper Summary:**

The paper proposes using non-gaussianity of the distribution produced by projecting residual-stream activations onto a direction as a signal for the direction’s interpretability. They show that this consistently identifies interpretable directions and suggest applying ICA to the residual stream in order to discover non-gaussian directions unsupervised. They demonstrate that a previous work which used ICA underestimated its interpretability due to sign ambiguity and once correcting for it, ICA produces more interpretable directions competitive with SAEs on multiple metrics.

**Presentation:** 3: Clear and easy to follow.

**Informativeness:** 2: I learned something interesting or useful.

**Soundness:** 2: I have minor or fixable concerns about the paper’s soundness.

**Critical Strengths:**

- The authors show that non-gaussianity is a strong signal for interpretable directions in activation space with a systematic evaluation including both human annotators and LLM based evaluation across 3 models. Their analysis goes beyond previously shown results for non-gaussianity [1] as a signal for interpretability, providing clear evidence of a correlation between the two.
- The utilized method (FastICA) is a simple and computationally efficient alternative to training SAEs for discovering a set of interpretable directions.

[1] Sparse Autoencoders Find Highly Interpretable Features in Language Models

**Critical Weaknesses:**

While the paper includes probe level interventions and some qualitative steering examples, the evaluations are mostly focused on showing that ICA directions are associated with interpretable concepts or contain concept-relevant information. While the TPP evaluation measures the effect of perturbations, it captures the effect on probes rather than directly on model behavior. In order to get a better sense of how well ICA holds in common featurizer applications, a more systematic causal evaluation such as MIB [2] or AxBench [3] would provide stronger evidence that ICA localizes features well. It would also provide a clearer comparison against commonly utilized featurizers and show what advantages the method has.

[2] MIB: A Mechanistic Interpretability Benchmark

[3] AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders

**Questions For Authors:**

- Figure 4, side B seems to contain only two curves: SAE and stronger-tail sign. Do you have the results for the arbitrary sign ICA as well? It would help make the claim that sign ambiguity was the problem. Similarly, do you have the results for the other models?
- How many annotators participated in the human eval and if multiple, could you report inter-annotator agreement ? It would help assess the reliability of the scores.

**Flag For Ethics Review:** No

**Rating:** 2: Weak rejection

**Confidence:** 4: I have substantial direct research experience on this topic.



## Rebuttals DSqt

We thank the reviewer for recognizing our evidence across three models that non-Gaussianity is a useful signal for discovering interpretable directions, as well as the simplicity and computational efficiency of FastICA.

**W. Causal evaluation on generated answers.** 

We added RAVEL [1], a task included in SAEBench [2] and MIB [3], to test whether ICA supports selective changes to model-generated answers. Following SAEBench’s RAVEL protocol, we learn a mask for each target attribute and replace the selected coordinates of a base entity with those of a source entity. For example, transferring Tokyo’s continent to Paris should change the continent answer from Europe to Asia while preserving France as the country. **Cause** measures successful transfer of the target attribute. **Isolation** measures preservation of other tested attributes. **Disentangle** averages the two.

On Gemma 2 2B layer 12, ICA achieves a Disentangle score of **0.7671**, compared with **0.7208** for PCA and **0.6635** for the Gemma Scope 16K SAE, averaged over six city and Nobel laureate attribute tasks and two evaluation seeds. ICA also improves both mean Cause and mean Isolation over both baselines. We added the results to Section 2.3.4 and Figure 10, with the protocol in Appendix E.5.

[1] Huang et al. (2024). [RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations](https://aclanthology.org/2024.acl-long.470/).
[2] Karvonen et al. (2025). [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](https://arxiv.org/abs/2503.09532).
[3] Mueller et al. (2025). [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151).

**Q1. Arbitrary-sign ICA in Figure 4B and results on other models.** 

We added the arbitrary-sign ICA curve for Qwen 3.5 9B Base to Figure 4B. Using the same fitted directions, stronger-tail orientation improves mean automated interpretability over arbitrary signs at all four evaluated layers. We also compared stronger-tail ICA with the Gemma Scope 16K SAE at four Gemma 2 2B layers. Each method evaluates 150 features per layer. 

| Layer | ICA (stronger-tail sign) | SAE (Gemma Scope 16K) |
| ----- | ------------------------ | --------------------- |
| 5     | 0.482                    | **0.510**             |
| 12    | 0.367                    | **0.397**             |
| 18    | **0.500**                | 0.406                 |
| 25    | **0.353**                | 0.350                 |

**Q2. Number of annotators and inter-annotator agreement.** 

The original submission reported ratings from one annotator. A second expert independently annotated the same 150 components using the same 0–5 rubric, during the rebuttal phrase. The second annotator was blinded to the original labels and ratings, the component statistics (kurtosis, skewness, and logcosh), and the component groupings.

| Metric                              | GPT-2 Small | Gemma 2 2B  | Qwen 3.5 9B Base | Overall                  |
| ----------------------------------- | ----------- | ----------- | ---------------- | ------------------------ |
| Components annotated                | 50          | 50          | 50               | 150                      |
| High confidence, A1                 | 41/50 (82%) | 42/50 (84%) | 40/50 (80%)      | 123/150 (82.0%)          |
| High confidence, A2                 | 46/50 (92%) | 43/50 (86%) | 45/50 (90%)      | 134/150 (89.3%)          |
| Exact score agreement               | 76%         | 82%         | 60%              | 72.7%                    |
| Scores within one point             | 90%         | 100%        | 90%              | 93.3%                    |
| Agreement on high-confidence status | 90%         | 98%         | 90%              | 92.7%                    |
| Quadratically weighted Cohen’s κ    | 0.667       | 0.958       | 0.702            | **0.772 [0.635, 0.874]** |

Scores of 4 or 5 count as high confidence. The mean high-confidence rate across annotators is **85.7%**, and both annotators rate **123/150 components (82.0%)** as high confidence. Brackets give the 95% paired bootstrap confidence interval. The revised Appendix C.2 includes the protocol, both annotators’ labels and scores for all 150 components, and examples of disagreements.

We believe these additions address your concerns and questions.



## **Official Review of Submission3245 by Reviewer piFb**

Official Reviewby Reviewer piFb22 Oct 2026, 05:53 (modified: 06 Nov 2026, 07:15)Everyone[Revisions](https://openreview.net/revisions?id=87IOQIjJqL)

**Paper Summary:**

This paper investigates non-Gaussianity as a statistical signature of interpretable directions in language model activations and introduces ICA Lens, a lightweight framework based on FastICA. It also identifies and corrects a sign-related bias in previous ICA interpretability evaluations. Experiments across three language models show competitive performance against SAEs in interpretability, sparse reconstruction, sparse probing, and targeted probe perturbation, with substantially lower parameter memory. However, the independent contribution of non-Gaussianity, generation-time steering effectiveness, and comparison fairness require further investigation.

**Presentation:** 2: Generally understandable, but requires substantial effort in places.

**Informativeness:** 2: I learned something interesting or useful.

**Soundness:** 2: I have minor or fixable concerns about the paper’s soundness.

**Critical Strengths:**

1. The paper presents comprehensive experiments cover GPT-2 Small, Gemma 2 2B, and Qwen 3.5 9B Base, combining human annotation, automated interpretability evaluation, and statistical analysis. Among 150 randomly sampled components, 123 receive high-confidence human labels. Sparse reconstruction, sparse probing, and targeted probe perturbation further assess practical utility beyond selected high-activation examples.
2. The proposed method provides a lightweight alternative to SAEs. ICA fits a complete basis in approximately 12–13 minutes per layer for the 9B model and uses less parameter memory than matched public SAEs, while remaining competitive in sparse reconstruction and concept prediction.
3. The analysis shows that both SAE decoder directions and concept-derived directions exhibit substantial non-Gaussianity, motivating direct feature discovery through ICA. Identifying and correcting the sign-sensitive evaluation bias is also a valuable methodological contribution.

**Critical Weaknesses:**

1. The contribution of non-Gaussianity is not fully isolated. High kurtosis may also arise from rare tokens, positional markers, code formatting, or a few extreme outliers. The experiments do not sufficiently distinguish non-Gaussianity from correlated factors such as activation sparsity and concept frequency. The component-level Spearman correlation of ρ = 0.45 supports an association but does not resolve these potential confounds.
2. Evidence for generation-time control remains limited. TPP measures changes in target and non-target probe accuracy after ablation, directly supporting selective intervention at the probe level. It does not establish that the model selectively uses the corresponding concepts during generation.
3. Matched feature counts may not ensure comparable interventions. ICA produces exactly (d) components, whereas SAEs use larger, overcomplete dictionaries. Although the comparisons match the top-k feature budget, equal feature counts need not imply equal activation sparsity or perturbation magnitude. Additional comparisons controlling for these factors would strengthen the conclusions.
4. I suggest moving some example component interpretations from appendix D to Section 2.3.

**Questions For Authors:**

1. Does the correlation between non-Gaussianity and interpretability remain after controlling for activation sparsity and concept frequency in the text?
2. Can the author reports intervention success rates and effects on unrelated capabilities, beyond probe-level TPP results?
3. Do ICA’s advantages over SAEs persist when comparisons match perturbation magnitude or actual activation sparsity, rather than only the number of selected features?

**Flag For Ethics Review:** No

**Rating:** 3: Weak acceptance

**Confidence:** 2: I have broad familiarity with this field but little or no direct research experience on this topic.



## Rebuttals piFb

Thank you for highlighting the breadth of our evaluation across three models, the efficiency of ICA Lens, and the importance of correcting the sign bias in prior evaluations. 

**W1. The contribution of non-Gaussianity is not fully isolated**

Our claim is that non-Gaussianity is associated with interpretability and is useful for feature discovery, **not that it causes interpretability**. The toy example in Appendix A.1 illustrates how extreme observations can drive non-Gaussianity. In this example, related tokens form a coherent tail. Removing those tail observations would leave a background distribution closer to Gaussian. The positional markers and code-formatting patterns you mention can themselves be interpretable if they hold up under counterfactual testing.

**W1/Q1. controlling the input data**

In Appendix B.5, we added a controlled fitting experiment with 1K, 5K, 10K, 100K, and 1M token activations. We expect larger fitting datasets to contain more occurrences of recurring concepts. With 1K activations, median excess kurtosis reached 514.94 on the fitting tokens but only 0.53 on held-out documents. With 100K and 1M activations, held-out medians rose to 11.04 and 11.83. The results suggest that ICA needs enough fitting activations to recover directions whose non-Gaussianity persists on unseen documents. One explanation is that larger samples cover more concepts and provide more examples of each concept across contexts, helping ICA recover recurring activation patterns.

As in SAE-based feature discovery, we use activations from pretraining text to discover interpretable features. Our experiments show that ICA works well for feature discovery in this setting. This experiment provides an initial test of how the amount of input data/concept affects the recovered non-Gaussian signal. **We believe that fine-grained control of activation sparsity and concept frequency is beyond the scope of this paper and deserves future research, we have add this to our future work.**

To examine whether this relationship generalizes beyond fitted ICA components, we also analyze **native MLP neurons, which do not require a separate feature-fitting procedure**. In Appendix A.2, we added an exploratory analysis of native MLP neurons learned during pretraining. We find that excess kurtosis is positively associated with lexical concentration in their activation tails. This extends our analysis beyond ICA and suggests that **non-Gaussianity may also help identify native MLP neurons worth interpreting**.

**W2/Q2. Evidence for generation-time control remains limited.**

We added RAVEL [1], a task included in SAEBench [2] and MIB [3], to test whether ICA supports selective changes to model-generated answers. Following SAEBench’s RAVEL protocol, we learn a mask for each target attribute and replace the selected coordinates of a base entity with those of a source entity. For example, transferring Tokyo’s continent to Paris should change the continent answer from Europe to Asia while preserving France as the country. **Cause** measures successful transfer of the target attribute. **Isolation** measures preservation of other tested attributes. **Disentangle** averages the two.

On Gemma 2 2B layer 12, ICA achieves a Disentangle score of **0.7671**, compared with **0.7208** for PCA and **0.6635** for the Gemma Scope 16K SAE, averaged over six city and Nobel laureate attribute tasks and two evaluation seeds. ICA also improves both mean Cause and mean Isolation over both baselines. We added the results to Section 2.3.4 and Figure 10, with the protocol in Appendix E.5.

[1] Huang et al. (2024). [RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations](https://aclanthology.org/2024.acl-long.470/).
[2] Karvonen et al. (2025). [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](https://arxiv.org/abs/2503.09532).
[3] Mueller et al. (2025). [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151).

**W3/Q3. Matched feature counts may not ensure comparable interventions.**

Our paper includes the following empirical comparisons:

1. In Appendix E.3, we have an experiment controlling for SAE dictionary size. We included compact Matryoshka SAEs with dictionary sizes of 128 and 512, both smaller than ICA's basis size. Under the same feature budgets, these smaller SAEs still do not outperform ICA in sparse probing. **This suggests that simply reducing SAE dictionary size does not close the performance gap under the same budgets.**
2. We also examine this from our existing results. For example, in Figure 9, SAE top-100 corresponds to roughly ICA top-3, top-14, and top-6 across the three models when matching the fraction of dictionary features selected. The TPP score gaps are relatively modest in several settings. **This suggests that ICA remains competitive even when accounting for dictionary size differences; ICA can approach SAE's intervention performance using substantially fewer components.**

We believe that comparisons should consider practical interpretability scenarios:

1. **Matching by proportion would reward dictionary size rather than explanation quality.** A larger dictionary would receive a larger absolute feature budget, making it difficult to compare how much useful information each feature captures.
2. **Human attention is limited.** In real interpretability debugging, if an SAE needs 100 features to capture concepts that ICA can capture with just 5 components, it becomes much harder for researchers to track and understand those features and new activations. This is why we believe matching the absolute feature budget is meaningful for practical interpretability.
3. Finally, we want to emphasize the intended role of ICA Lens. SAEs can provide more numerous and fine-grained features, and we do not claim that ICA covers the same number of unique concepts. **ICA Lens is meant to be a lightweight first lens**, helping researchers quickly identify promising layers and directions before investing substantial time and engineering effort into training additional SAEs for more detailed analysis.



## **Official Review of Submission3245 by Reviewer zuSC**

Official Reviewby Reviewer zuSC21 Oct 2026, 18:17 (modified: 06 Nov 2026, 07:15)Everyone[Revisions](https://openreview.net/revisions?id=QTkNKOAopz)

**Paper Summary:**

This paper focuses on the problem of finding interpretable directions in language models, which has been addressed in the past with several methods including sparse autoencoders (SAE), principal component analysis (PCA) and independent component analysis (ICA). Through analysis experiments, they find that the projection distributions of these interpretable directions from above methods are more non-Gaussian than random directions. Hence, they propose to discover these directions by directly searching for non-Gaussian directions using FastICA instead of training sparse dictionaries like in SAEs. They also find a correction is required to the FastICA components to make sure the more interpretable directions are chosen. Through extensive experiments, they show comparable performance to SAEs while being more efficient and using less memory.

**Presentation:** 2: Generally understandable, but requires substantial effort in places.

**Informativeness:** 2: I learned something interesting or useful.

**Soundness:** 2: I have minor or fixable concerns about the paper’s soundness.

**Critical Strengths:**

- The paper is quite interesting and generally easy to follow. It focuses on ICA as a faster initial interpretability analysis step compared to more expensive SAE training, which will be valuable to the community.
- Evaluating the non-Gaussianity of SAE directions as well as PCA on LLM activations is interesting and seems to be novel.

**Critical Weaknesses:**

- The comparison between ICA and SAEs does not seem completely fair to me. Like ICA can only discover (upto) d components while SAEs can discover k \times d (with k as the SAE expansion factor). Then, even if ICA and SAEs have comparable average performance on different metrics, the total number of useful components found in SAEs will be (roughly) k times more than in ICA. In that sense, ICA is k times less useful than an SAE but I do concede, also k times more efficient than an SAE (which the paper already does claim and I agree with it).
  - Now, there are also other problems with SAEs like feature absorption [W1] and that not all SAE features are useful [W2, W3] which weakens my argument. But I feel that the current evaluation is on 150 features randomly sampled with matched eligibility filters, but it does not quantify the overall/total interpretability of SAE or ICA, it only quantifies the average on a small set.

[W1] Chanin et al., "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders", NeurIPS 2025

[W2] Arad et al., "SAEs Are Good for Steering – If You Select the Right Features", EMNLP 2025

[W3] Kulkarni et al., "Interpretable and Steerable Concept Bottleneck Sparse Autoencoders", CVPR 2026

**Questions For Authors:**

- Please address the weaknesses section.

- The full term "Independent Component Analysis" is first used on Page 9 but the acronym ICA is used from Page 1 itself. I suggest mentioning the full term on Page 1 so it is clear what ICA is.

**Flag For Ethics Review:** No

**Rating:** 3: Weak acceptance

**Confidence:** 3: I am familiar with this topic and its relevant literature.



## Rebuttals zuSC

Thank you for recognizing the novelty of our non-Gaussianity analysis and the practical value of ICA as a lightweight first step for interpretability.

**W1. ICA vs. SAE feature budgets**

Our paper includes the following empirical comparisons:

1. In Appendix E.3, we have an experiment controlling for SAE dictionary size. We included compact Matryoshka SAEs with dictionary sizes of 128 and 512, both smaller than ICA's basis size. Under the same feature budgets, these smaller SAEs still do not outperform ICA in sparse probing. **This suggests that simply reducing SAE dictionary size does not close the performance gap.**
2. We can also examine this from our existing results. For example, in Figure 9, SAE top-100 corresponds to roughly ICA top-3, top-14, and top-6 across the three models when matching the fraction of dictionary features selected. The TPP score gaps are relatively modest in several settings. **This suggests that ICA remains competitive even when accounting for dictionary size differences; ICA can approach SAE's intervention performance using substantially fewer components.**

We also believe that comparisons should consider practical interpretability scenarios:

1. **Matching by proportion would reward dictionary size rather than explanation quality.** A larger dictionary would receive a larger absolute feature budget, making it difficult to compare how much useful information each feature captures.
2. **Human attention is limited.** In real interpretability debugging, if an SAE needs 100 features to capture concepts that ICA can capture with just 5 components, it becomes much harder for researchers to track and understand those features and new activations. This is why we believe matching the absolute feature budget is meaningful for practical interpretability.
3. Finally, we want to emphasize the intended role of ICA Lens. SAEs can provide more numerous and fine-grained features, and we do not claim that ICA covers the same number of unique concepts. **ICA Lens is meant to be a lightweight first lens**, helping researchers quickly identify promising layers and directions before investing substantial time and engineering effort into training additional SAEs for more detailed analysis.

**W2. Number of evaluated features**

The original SAE paper [1] also relied on randomly sampled features rather than evaluating the entire dictionary. We reproduced and followed its evaluation protocol with 150 sampled features.

Moreover, evaluating interpretability at scale is expensive. Even with random sampling, we spent over $1,500 in API costs on our experiments, and each human annotator spent several hours labeling components. Across our randomly sampled components, an average of 85.7% received high-confidence interpretations from the two annotators. We believe this provides meaningful evidence that ICA recovers interpretable components, although it does not establish the total number of distinct interpretable concepts.

We agree that a full-dictionary evaluation would be valuable, and we hope to collaborate on scaling up the annotation effort in future work.

[1] Huben et al. (2024). [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/pdf/2309.08600).

**Q2. ICA acronym**

Thank you for catching this. We have added **independent component analysis (ICA)** at its first mention on Page 1.





## Author Summary: Key Revisions, Scientific Contributions, and Broader Impact

We thank all reviewers for their thoughtful feedback and for recognizing the novelty of our non-Gaussianity analysis, the importance of correcting the prior ICA evaluation bias, and the practical efficiency of ICA Lens.

We have substantially strengthened the paper during the rebuttal period, particularly through new causal evaluations, independent human annotations, and additional analyses addressing the reviewers' concerns. Below, we summarize the key updates and clarify what we believe are the broader contributions of this work.

**1. Key Concerns and New Evidence**

**Causal interventions beyond probing (DSqt, piFb).** We added RAVEL, a benchmark used in SAEBench and MIB, to evaluate selective control over model-generated answers. On Gemma 2 2B, ICA achieves a **Disentangle score of 0.767**, outperforming PCA (0.721) and Gemma Scope SAE (0.664) across six attribute tasks and two seeds. ICA also achieves higher mean Cause and Isolation scores than both baselines. This provides direct evidence of selective interventions on model outputs, beyond probe-level TPP.

**Human annotation reliability (DSqt, zuSC).** We added a second expert who independently annotated the same 150 randomly sampled components. The two annotators achieve **92.7% agreement on high-confidence status**, with a quadratically weighted Cohen's κ of **0.772**. Both annotators assign high confidence to **123/150 components (82.0%)**, supporting the reliability of our interpretability findings.

**Non-Gaussianity and generalization (piFb).** We added a fitting-data ablation from 1K to 1M tokens. With only 1K fitting activations, median excess kurtosis reaches 514.94 on training data but drops to 0.53 on held-out documents. With 1M activations, the held-out median rises to 11.83, suggesting that sufficient data helps ICA recover non-Gaussian structure that generalizes. We also extended our analysis to native MLP neurons, finding a positive association between excess kurtosis and lexical concentration (Spearman ρ = 0.55).

**ICA–SAE comparison and feature budgets (piFb, zuSC).** We added compact Matryoshka SAE baselines with dictionary sizes of 128 and 512. Under matched sparse-probing budgets, ICA still outperforms these smaller dictionaries. We also clarify that matching feature counts measures utility under a limited analysis budget, rather than total concept coverage. We do not claim that ICA discovers as many distinct concepts as overcomplete SAEs.

**Sign-sensitive evaluation (DSqt).** We added the missing arbitrary-sign ICA results for Qwen 3.5 9B and additional comparisons with Gemma Scope SAEs, further supporting our finding that the arbitrary sign of ICA components contributed to their previously underestimated interpretability.

**2. Why We Believe This Work Matters**

We believe the contribution of this paper goes beyond showing that ICA can compete with SAEs. It addresses a more fundamental question: **what statistical structure makes a direction interpretable, and can we use that structure directly for feature discovery?**

Our work contributes at several levels.

**Scientific insight.** Rather than learning features through an indirect sparse-reconstruction objective and evaluating interpretability afterward, we show that non-Gaussianity itself provides a useful unsupervised signal for discovering interpretable directions. This offers a different perspective on the structure already present in language-model activations.

**Correcting a prior evaluation bias.** An early ICA–SAE comparison reported substantially lower interpretability for ICA. We identify a sign-sensitive evaluation bias and show that correcting it recovers much of the reported gap, without changing the fitted ICA directions. We believe revisiting such established conclusions is important for developing reliable interpretability methods and benchmarks.

**A practical first lens.** ICA Lens provides an efficient way to inspect models before committing to expensive dictionary learning. For Qwen 3.5 9B, we fit a complete ICA basis in approximately **12–13 minutes per layer on a single NVIDIA GB10**, while using **7–43× less parameter memory** than the matched public SAEs across our three models. This makes repeated analysis across models, layers, and checkpoints more practical.

**Complementing, rather than replacing, SAEs.** SAEs remain valuable for discovering larger and more fine-grained feature dictionaries. Our goal is different: to recover useful interpretable structure quickly, helping researchers decide where more detailed analysis is worth the additional computational and engineering cost. In real interpretability workflows, the number of features a researcher can inspect and reason about is also a meaningful constraint.

**3. Summary of Revisions**

For convenience, the main additions to the revised manuscript are:

- **Section 2.3.4 / Figure 10 / Appendix E.5:** New RAVEL causal intervention evaluation on generated answers.
- **Section 2.2 / Appendix C.2:** Second independent human annotator, inter-annotator agreement, and complete annotations.
- **Figure 4 / Appendix C.4:** Expanded sign-correction comparisons across Qwen and Gemma.
- **Appendix B.5:** Fitting-data scaling and held-out non-Gaussianity analysis.
- **Appendix A.2:** Non-Gaussianity analysis of native MLP neurons.
- **Appendix E.3:** Compact Matryoshka SAE baselines and additional sparse-probing comparisons.

Taken together, we believe ICA Lens makes **three complementary contributions: a scientific principle for discovering interpretable structure, a correction to an influential evaluation of ICA, and an efficient, practical workflow for model interpretability**.

We hope these additional experiments and clarifications help address the reviewers' concerns and make the paper's contributions clearer. We thank the reviewers and ACs again for their time and consideration.





第一部分：分类汇总审稿人的疑惑+我们的解决（突出重点+结果）

第二部分：解释为什么我们的研究是重要的、对领域的影响（可以引用下面的table）

- 我们的贡献涵盖了：scientific contribution、修正领域错误认知、补全领域评估gaps、实践价值

| Prior work and gaps                                          | Our findings and contributions                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| SAEs discover interpretable directions through sparse reconstruction (Huben et al., 2024; Gao et al., 2025), but it remains unclear what statistical properties distinguish interpretable directions and whether these properties can directly guide feature discovery. | **Discovery Principle.** We propose non-Gaussianity as an inductive bias for interpretable feature discovery, showing that interpretable directions can be recovered by directly searching for non-Gaussian projections in activation space. |
| An early ICA–SAE comparison reported low interpretability for ICA (Huben et al., 2024). ICA's potential for LLM interpretability has long been underestimated. | **Evaluation Correction.** We identify and correct a sign-sensitive evaluation bias, recovering much of the reported gap and making ICA competitive with the evaluated SAEs without changing the fitted directions. |
| Earlier work found only a weak positive correlation between the kurtosis of SAE latent activations and automated interpretability (Huben et al., 2024), and did not examine this relationship for ICA directions. | **Interpretability Evidence.** We systematically examine the relationship between non-Gaussianity and interpretability across ICA fitting checkpoints and components, finding a clearer positive association. |
| Training separate SAE dictionaries across models and layers can be costly (Lieberum et al., 2024), while prior ICA studies (Yamagiwa et al., 2023; Musil & Mareček, 2024) lacked a practical workflow for modern LLM interpretability. | **Efficient Workflow.** We release ICA Lens, to our knowledge the first ICA-focused, end-to-end workflow for LLM interpretability, combining GPU-accelerated fitting, better convergence diagnostics, systematic annotation, and faithful evaluation. It provides a lightweight first lens for understanding models. |
| ICA remains underrepresented in widely used SAE benchmarks (Karvonen et al., 2025) and feature-visualization platforms (Neuronpedia), leaving its practical potential insufficiently explored. | **Downstream Utility.** We compare ICA with public SAEs on interpretability, sparse reconstruction, sparse probing, probe perturbation, and causal interventions, and illustrate its use for activation inspection and steering. |

第三部分：汇总对论文的修改和额外补充（说清楚补充了什么、意义-从AC视角理解价值、论文对应修改的地方）

例如

实验补充：

- 额外的人工标注和分析聚合：展现可靠性
- 不同输入数据规模fitting的分析
- MLP neurons：展现非高斯性-可解释性更general的扩展

效率换算补充：we fitted our ICA on a single NVIDIA GB10 with 128 GB of unified memory (≈0.08 H100, 0.13 A100). it took 12 min per layer for the Qwen3.5-9B model. it might take just 1-5 min/layer on an A100/H100.





Importance：解释为什么我们的研究是重要的

- 引入新的视角（scientific）
- first lens
- 修正的错误认知

Impact：对领域的影响

- 实践价值
- 评估价值
- 研究价值




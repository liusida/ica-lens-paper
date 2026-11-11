# ICA Lens Rebuttals

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


## Summary: Key Revisions, Scientific Contributions, and Broader Impact

We thank all reviewers for their thoughtful feedback and for recognizing the novelty of our non-Gaussianity analysis, the importance of correcting the prior ICA evaluation bias, and the practical efficiency of ICA Lens.

We have strengthened the paper during the rebuttal period through new causal evaluations, independent human annotations, and additional analyses addressing the reviewers' concerns. Below, we summarize the key updates and broader contributions of this work.

**Key Concerns and New Evidence**

**Causal interventions beyond probing (DSqt, piFb).** We added RAVEL, a benchmark used in SAEBench and MIB, to evaluate selective control over model-generated answers. On Gemma 2 2B, ICA achieves a **Disentangle score of 0.767**, outperforming PCA (0.721) and Gemma Scope SAE (0.664) across six attribute tasks and two seeds. ICA also achieves higher mean Cause and Isolation scores than both baselines. This provides direct evidence of selective interventions on model outputs, beyond probe-level TPP.

**Human annotation reliability (DSqt, zuSC).** We added a second expert who independently annotated the same 150 randomly sampled components. The two annotators achieve **92.7% agreement on high-confidence status**, with a quadratically weighted Cohen's κ of **0.772**. Both annotators assign high confidence to **123/150 components (82.0%)**, supporting the reliability of our interpretability findings.

**Non-Gaussianity and generalization (piFb).** We added a fitting-data ablation from 1K to 1M tokens. With only 1K fitting activations, median excess kurtosis reaches 514.94 on training data but drops to 0.53 on held-out documents. With 1M activations, the held-out median rises to 11.83, suggesting that sufficient data helps ICA recover non-Gaussian structure that generalizes. We also extended our analysis to native MLP neurons, finding a positive association between excess kurtosis and lexical concentration (Spearman ρ = 0.55).

**ICA–SAE comparison and feature budgets (piFb, zuSC).** We added compact Matryoshka SAE baselines (128/512 features), which ICA still outperforms under matched sparse-probing budgets.  A comparison using Figure 9 also suggests that ICA remains competitive on TPP when matching dictionary proportions, using substantially fewer components. Matching absolute feature counts is meaningful because human attention is limited, and proportional matching favors larger dictionaries. We do not claim equal concept coverage. Instead, ICA Lens serves as a lightweight first lens, helping researchers identify promising layers and directions before investing in more expensive SAE training.

**Sign-sensitive evaluation (DSqt).** We added the missing arbitrary-sign ICA results for Qwen 3.5 9B and additional comparisons with Gemma Scope SAEs, further supporting our finding that the arbitrary sign of ICA components contributed to their previously underestimated interpretability.

For convenience, the main additions to the revised manuscript are:

- **Section 2.1:** Clarified the practical bottlenecks of applying ICA to modern LLM interpretability, including slow CPU-based implementations and limitations of conventional convergence criteria, while better highlighting our algorithmic and methodological contributions.
- **Section 2.3.4 / Figure 10 / Appendix E.5:** New RAVEL causal intervention evaluation on generated answers.
- **Section 2.2 / Appendix C.2:** Second independent human annotator, inter-annotator agreement, and complete annotations.
- **Figure 4:** Expanded sign-correction comparisons across Qwen.
- **Appendix B.5:** Fitting-data scaling and held-out non-Gaussianity analysis.
- **Appendix A.2:** Non-Gaussianity analysis of native MLP neurons.
- **Appendix E.3:** Compact Matryoshka SAE baselines and additional sparse-probing comparisons.

**Why We Believe This Work Matters**

We summarize the key gaps in prior work and our contributions below.

| Prior work and gaps                                          | Our findings and contributions                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Scientific Gap.** SAEs discover interpretable directions through sparse reconstruction (Huben et al., 2024; Gao et al., 2025), but it remains unclear what statistical properties distinguish interpretable directions and whether these properties can directly guide feature discovery. | **Discovery Principle.** We propose non-Gaussianity as an inductive bias for interpretable feature discovery, showing that interpretable directions can be recovered by directly searching for non-Gaussian projections in activation space. |
| **Evaluation Gap.** An early ICA–SAE comparison reported low interpretability for ICA (Huben et al., 2024). ICA's potential for LLM interpretability has long been underestimated. | **Evaluation Correction.** We identify and correct a sign-sensitive evaluation bias, recovering much of the reported gap and making ICA competitive with the evaluated SAEs without changing the fitted directions. |
| **Analysis Gap.** Earlier work found only a weak positive correlation between the kurtosis of SAE latent activations and automated interpretability (Huben et al., 2024), and did not examine this relationship for ICA directions. | **Interpretability Evidence.** We systematically examine the relationship between non-Gaussianity and interpretability across ICA fitting checkpoints and components, finding a clearer positive association. |
| **Engineering Gap.** Training separate SAE dictionaries across models and layers can be costly (Lieberum et al., 2024), while prior ICA studies (Yamagiwa et al., 2023; Musil & Mareček, 2024) did not address practical challenges at LLM scale, including slow CPU-based fitting and convergence criteria not tailored to interpretability analysis. | **Efficient Workflow.** We address these challenges and release ICA Lens, to our knowledge the first end-to-end workflow combining GPU-accelerated ICA fitting, improved convergence diagnostics, systematic annotation, and faithful evaluation. **Note**: We fitted our ICA on a single NVIDIA GB10 with 128 GB of unified memory (≈0.08 H100, 0.13 A100). it took 12 min per layer for the Qwen3.5-9B model. it might take just 1-5 min/layer on an A100/H100. |
| **Applications Gap.** ICA remains underrepresented in widely used SAE benchmarks (Karvonen et al., 2025) and feature-visualization platforms (Neuronpedia), leaving its practical potential insufficiently explored. | **Downstream Utility.** We compare ICA with public SAEs on interpretability, sparse reconstruction, sparse probing, probe perturbation, and causal interventions, and illustrate its use for activation inspection and steering. |

We believe ICA Lens corrects a long-standing misconception in the interpretability community, offers a fresh perspective, and opens up promising directions for future research that could attract broader community interest. For example, with this lightweight first lens, we **(especially independent interp researchers who don't have too much GPUs)** can now explore the following directions much more freely:

1. **Understanding how interpretable features emerge during training.** Because ICA Lens can be refitted efficiently, researchers could track residual directions across pretraining, post-training, and RL checkpoints to study when features emerge, strengthen, change meaning, or disappear, without training a new SAE dictionary at every checkpoint.
2. **Exploring interpretability beyond individual residual-stream layers.** Non-Gaussian feature discovery could extend to attention-head outputs, MLP activations, and joint representations across layers, potentially revealing how interpretable structure develops and transforms throughout a model.
3. **Extending feature discovery to reasoning and agents.** Rather than analyzing only individual tokens, researchers could examine representations at the reasoning-step or trajectory level, searching for directions associated with planning, uncertainty, recovery, or failure. Longer term, similar analyses could explore shared structure across language, vision, and action representations.
4. Extend the analysis to language, vision, and action representations to investigate potential shared latent factors.

We hope ICA Lens encourages further research into these directions and inspires the community to revisit other simple but potentially overlooked approaches to understanding language models.

We thank the reviewers and ACs again for their time and consideration.

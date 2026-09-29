# English-to-target language steering

Minimal SAE/ICA language-steering comparison based on
[Causal Language Control in Multilingual Transformers via Sparse Feature Steering](https://arxiv.org/abs/2507.13410).

- **Model:** `google/gemma-2-2b`
- **Layer:** 20
- **Calibration pairs:** 1000 per language
- **Final-token mode:** `text`
- **Steering:** `all-positions`
- **Evaluator:** `gpt-4.1-mini-2025-04-14`

## Selection

Evaluate each of the three largest activation contrasts on four prompts. Select
the candidate with the most passing outputs.

| Language | SAE feature | SAE offset | ICA component | ICA offset | Signed cosine |
|---|---:|---:|---:|---:|---:|
| Chinese | F6360 | +83.7958 | C105 | +21.4227 | 0.7896 |
| French | F12332 | +100.3192 | C25 | +23.0798 | 0.8956 |
| Japanese | F4945 | +91.6278 | C145 | -23.3301 | 0.6853 |
| Spanish | F8590 | +91.8189 | C42 | +17.2218 | 0.8537 |

## Generations

For every selected feature and component, show the generation from the
shared prompt *Artificial intelligence is important because*.

<table style="width: 100%; table-layout: fixed;">
<colgroup>
<col style="width: 10%;">
<col style="width: 45%;">
<col style="width: 45%;">
</colgroup>
<thead><tr><th>Language</th><th>SAE</th><th>ICA</th></tr></thead>
<tbody>
<tr><td style="vertical-align: top;">Chinese</td><td style="vertical-align: top;"><strong>F6360</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes it possible for machines to学习，思考和做出决策。人工智能技术可以被应用于各种不同的领域，包括医疗保健，农业，能源和环境等。人工智能可以帮助人类解决一些棘手的问题，包括疾病的诊断和</td><td style="vertical-align: top;"><strong>C105</strong><br><span style="color: #777;">Artificial intelligence is important because</span>它可以提高效率，减少错误，并提高个人和企业对信息的可用性。它可以帮助我们解决许多问题，并使生活更轻松。<br><br>人工智能可以帮助我们解决许多问题，例如提高效率，降低错误，并提高</td></tr>
<tr><td style="vertical-align: top;">French</td><td style="vertical-align: top;"><strong>F12332</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes it possible pour l&#x27;intelligence artificielle, l&#x27;intelligence artificielle peut être appliquée à des domaines divers et variés, tels que l&#x27;apprentissage automatique, l&#x27;intelligence des machines, le traitement de l&#x27;image et du son,</td><td style="vertical-align: top;"><strong>C25</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes labo&lt;strong&gt;ur&lt;/strong&gt; &lt;strong&gt;moins&lt;/strong&gt; &lt;strong&gt;chère&lt;/strong&gt;.<br><br>L&#x27;intelligence artificielle est importante parce qu&#x27;elle rend les &lt;strong&gt;travailles moins chères&lt;/strong&gt;.<br><br>L&#x27;intelligence artificielle est une technologie</td></tr>
<tr><td style="vertical-align: top;">Japanese</td><td style="vertical-align: top;"><strong>F4945</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes it possible forコンピューター to do things that人間 cannot.<br>It enablesコンピューター to learn fromデータ and to判断 what is正しい and what is誤り.<br>It also enablesコンピューター todo things that人間 cannot, such as</td><td style="vertical-align: top;"><strong>C145</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes it possible for machines to learn from data and perform tasks that人はできない。<br><br>AIは、機械がデータを学習し、人はできないことをできるようになるため、重要な役割を果たしています。<br><br>AIは、人の考えや行動を再現する</td></tr>
<tr><td style="vertical-align: top;">Spanish</td><td style="vertical-align: top;"><strong>F8590</strong><br><span style="color: #777;">Artificial intelligence is important because</span> it makes nuestra vida más fácil y nos ayuda en muchas cosas. A veces, los programas de inteligencia artificial se ven como algo mágico, pero el mundo real es mucho más sencillo de lo que parece.<br><br>En este artículo, aprenderá todo sobre inteligencia</td><td style="vertical-align: top;"><strong>C42</strong><br><span style="color: #777;">Artificial intelligence is important because</span> de-humanización de las tareas y la falta de empatía en la interacción con el usuario son problemas que la IA puede resolver.<br><br>La inteligencia artificial (IA) es la imitación de la capacidad de un sistema para pensar, aprender, razon</td></tr>
</tbody>
</table>

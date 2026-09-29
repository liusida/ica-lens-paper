# Steering

ICA coordinates can be edited before they are mapped back into the residual
stream. This page reproduces a qualitative intervention in which Qwen 3.5 2B
changes its answer from **Quantum Computing** to **Neuroplasticity**.

## Run the example

Run the following cells in order in Jupyter or Colab.

### 1. Generate the baseline

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

On the tested model revision, greedy decoding starts with **Quantum
Computing**.

### 2. Inspect the component and its selected tail

First read the tail selected from profile-wide score skewness:

```python
component = lens.component_profile(layer=5, component=188)
print(component["tail_direction"])
print(component["sign_statistics"])
print(component["examples"][component["tail_direction"]]["tokens"][:10])
```

For this artifact, `C188` has selected tail `negative`. The profile chooses the
side indicated by population skewness and provides examples from that side.
It does not by itself prove that a particular concept belongs
to that direction, so verify the interpretation with independent probes.

Analyze concepts in separate inputs. Do not put them in one list: in a causal
language model, an earlier concept can affect every later token.

```python
neuroscience_result = lens.analyze("Neuroscience.", layer=5)
neuroscience_result
```

In another cell:

```python
quantum_result = lens.analyze("Quantum computing.", layer=5)
quantum_result
```

For this exact Lens at layer 5, `C188` is strongly negative on the `uro` and
`science` tokens of “Neuroscience” (approximately `-8.7` and `-14.9`), but is
near zero throughout “Quantum computing.” Independent probes such as “The
human brain.” and “Neuroplasticity.” show the same negative direction.

The negative sign is supported by both the stored profile and these probes; it
was not inferred from the word “Neuroscience.” ICA signs are arbitrary, so the
same component in another independently fitted Lens could use the opposite
convention. Always read the profile belonging to the exact Lens being used.

### 3. Inspect the baseline conversation

Append the generated answer and decompose the completed conversation:

```python
baseline_messages = [
    *messages,
    {"role": "assistant", "content": baseline},
]
baseline_result = lens.analyze(baseline_messages, layer=5)
baseline_result
```

### 4. Clamp and regenerate

Clamp `C188` to `-20` at every processed token position and every
autoregressive step:

```python
steered = lens.generate(
    messages,
    layer=5,
    clamp=(188, -20.0),
    max_new_tokens=16,
)
print("Steered:", steered)
```

In our test, the response starts with **Neuroplasticity**. The language-model
weights remain cached between calls.

### Add a score offset instead

Clamping sets an absolute coordinate value. Additive steering instead moves the
coordinate by a calibrated offset:

```python
steered = lens.generate(
    messages,
    layer=5,
    steer=(188, -12.0),
    steering_scope="current-position",
    max_new_tokens=16,
)
```

Here, `steer=(188, -12.0)` applies

$$
h \leftarrow h - 12 A_{:,188},
$$

where $A_{:,188}$ is the component's writing direction. The default
`current-position` scope edits the final position of the prompt during prefill
and the newly generated position at every decoding step. This is usually the
most economical intervention. With `steering_scope="all-positions"`, the same
offset is also added to every prompt position during prefill.

Additive steering requires a Lens fitted without row normalization or a
preprocessing center. `clamp` and `steer` are mutually exclusive.

### 5. Inspect the resulting conversation

```python
steered_messages = [
    *messages,
    {"role": "assistant", "content": steered},
]
steered_result = lens.analyze(steered_messages, layer=5)
steered_result
```

This final call is an ordinary, unsteered analysis of the text produced by the
intervention. It shows how the original model naturally represents the
resulting conversation; it does not replay or trace the clamp used during
generation.

Generation is greedy by default. Exact continuations can change with the
model, Lens, Transformers version, hardware, prompt, or generation settings.

## Calibrate the target

An ICA score is a coordinate, not a semantic dosage. Larger magnitudes do not
guarantee a stronger or monotonic conceptual effect. A short sweep is more
informative than choosing an arbitrary value:

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

In our run, the answers progressed from Quantum Computing through Quantum
Biology and The Human Brain to Neuroplasticity. This sweep is empirical
calibration for one artifact and prompt, not a universal scale.

## How the clamp works

At the selected residual-stream layer, ICA Lens performs the following edit at
every processed token position and every autoregressive step:

1. L2-normalize the hidden state, as during ICA fitting.
2. Read its signed ICA scores.
3. Replace the selected coordinate with the target value.
4. Leave the other coordinates unchanged in score space.
5. Map the edited scores back through the writing matrix.
6. Restore the original residual-vector norm and return the edited vector to
   the model.

The `clamp=(188, -20.0)` argument requests

\[
s_{188} \leftarrow -20.
\]

Norm restoration rescales the reconstructed vector, so transforming the final
edited activation again may produce a value close to, rather than exactly,
`-20`.

The equivalent edit for an existing hidden-state tensor is:

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

`lens.generate()` installs and removes the model hook automatically for both
clamping and additive steering.

## Practical cautions

- Use signed scores, not energy shares, for steering.
- Treat component labels as hypotheses supported by examples, not built-in
  meanings or class labels.
- Read the stored tail direction, then confirm the concept-specific direction
  with independent probes; never infer it from the label.
- Match the model revision, layer, activation site, and preprocessing recorded
  by the fitted Lens.
- Calibrate moderate targets and compare against a deterministic baseline.
- Large targets can substantially rotate the residual vector and cause
  nonlinear or degraded behavior.
- A qualitative change does not show that one component deterministically
  controls a concept in every context.

The inverse mapping and norm restoration are described in
[Reconstruction](reconstruction.md).

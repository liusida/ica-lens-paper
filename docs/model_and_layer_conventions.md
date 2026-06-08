# Model and Layer Conventions

Layer names use zero-padded residual-stream names:

```text
layer_00
layer_01
...
```

The GPT-2 final residual-stream layer should refer to the raw post-block output
before the final layer norm, not `outputs.hidden_states[-1]` after `ln_f`.

Model short names:

```text
gpt2
gemma2_2b
qwen3_5_2b_base
```

# Text and chat

`lens.analyze()` accepts either raw text or a completed conversation. The input
form determines whether ICA Lens applies a chat template and which token scopes
are available.

## Raw text

Pass a string to analyze it exactly as ordinary model input:

```python
from icalens import ICALens

lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
result = lens.analyze(
    "The boat reached the bank before sunset.",
    layer=6,
)

result
```

No chat template, role marker, or conversation separator is added. Every token
produced from the string is captured and analyzed.

![Raw-text ICA Lens analysis](assets/text-analysis-notebook.png)

Passing a string to an instruction-tuned model also bypasses its chat template.
Use the conversation form when you want the same formatting the model uses for
chat.

## Conversations

Pass a list of messages with `role` and `content` fields:

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

Supported roles are `system`, `user`, and `assistant`. ICA Lens validates the
messages and renders them with the analyzed model tokenizer's chat template.
The resulting token sequence can contain:

- role and message-boundary markers;
- line breaks and separators;
- user, system, and assistant content;
- model-specific control tokens such as thinking markers.

The notebook result groups tokens by message while retaining the surrounding
template tokens associated with that message.

![Message-grouped ICA Lens analysis](assets/conversation-analysis-notebook.png)

## Token scopes

Conversation analysis defaults to `token_scope="all"`. Use a narrower scope
when template or other-role tokens are not part of the question you are asking.

| Scope | Tokens included |
| --- | --- |
| `"all"` | The entire rendered conversation, including template tokens |
| `"content"` | Message content from every role |
| `"user"` | User-message content only |
| `"assistant"` | Assistant-message content only |

For example:

```python
assistant_result = lens.analyze(
    messages,
    layer=16,
    token_scope="assistant",
)
```

Token scope changes which positions are returned; it does not change the
conversation passed through the model. The full rendered conversation remains
the context used to compute every selected activation.

Template boundaries are tokenizer-specific. ICA Lens aligns message content
through the fast tokenizer's character offsets, so chat analysis requires a
fast tokenizer with a chat template. If the selected scope contains no tokens,
`analyze()` raises an error rather than returning an empty result.

## Multi-turn conversations

Include the complete history in chronological order:

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

The result displays separate `System`, `User 1`, `Assistant 1`, `User 2`, and
`Assistant 2` token groups. Each activation is contextual: a token in a later
turn is computed after the model has processed all earlier turns.

## Analysis does not generate

`analyze()` captures activations for content that already exists. It does not
ask the language model to generate another assistant response.

To analyze generated text:

1. Generate a response with your preferred inference interface.
2. Append it to `messages` with role `assistant`.
3. Pass the completed conversation to `lens.analyze()`.

This separation keeps generation settings—such as sampling temperature and
maximum new tokens—independent from activation analysis.

## Context length and truncation

Use `context_length` to truncate the tokenized input before the model forward
pass:

```python
result = lens.analyze(
    messages,
    layer=16,
    context_length=2048,
)
```

For conversations, truncation is applied after the chat template is rendered.
Returned `result.positions` are positions in that rendered token sequence. A
message whose content falls entirely beyond the retained context does not
contribute tokens to the result.

## Inspect token alignment

The result keeps several aligned views of every selected position:

```python
result.tokens       # tokenizer vocabulary forms
result.token_texts  # individually decoded text
result.token_ids    # integer token IDs
result.positions    # positions in the model input sequence
result.scores       # signed ICA scores, one row per token
result.energy       # per-token component energy shares
```

All fields have the same leading token dimension. Some multilingual tokenizers
split one visible character across several byte fragments. The interactive
display marks an individually undecodable fragment as `<?>`; its tooltip shows
the token ID and the readable text obtained by decoding neighboring fragments
together.

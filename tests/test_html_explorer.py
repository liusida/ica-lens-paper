from __future__ import annotations

import torch

from icalens import AnalysisResult
from icalens.html import (
    VSCODE_NOTEBOOK_SCALE,
    _gpt2_byte_hex,
    _occurrence_target_span,
    analysis_iframe,
    write_explorer_html,
)


def test_occurrence_target_span_prefers_exact_offset_and_legacy_center() -> None:
    context = "such as sarcomas, a wide margin"
    target = "as"
    exact_start = context.index("as", context.index("as") + 1)

    assert _occurrence_target_span(
        {
            "context_target_start": exact_start,
            "context_target_end": exact_start + len(target),
        },
        context,
        target,
    ) == (exact_start, exact_start + len(target))
    assert _occurrence_target_span({}, context, target) == (
        exact_start,
        exact_start + len(target),
    )


def analysis_result() -> AnalysisResult:
    return AnalysisResult(
        tokens=("Ġhello",),
        token_texts=(" hello",),
        token_labels=(" hello",),
        token_tooltips=("Ġhello",),
        token_groups=(),
        token_ids=torch.tensor([1]),
        positions=torch.tensor([3]),
        activations=torch.tensor([[1.0, 2.0]]),
        scores=torch.tensor([[-2.0, 1.0]]),
        energy=torch.tensor([[0.8, 0.2]]),
        model="example/model@abc",
        layer=6,
        input_text="hello",
        token_scope="all text tokens",
        messages=(),
        component_profiles={
            0: {
                "dominant_sign": "negative",
                "tail_direction": "negative",
                "sign_statistics": {
                    "positive_fraction": 0.4,
                    "negative_fraction": 0.6,
                    "positive_energy_fraction": 0.2,
                    "negative_energy_fraction": 0.8,
                },
                "score_statistics": {
                    "skewness": -1.25,
                    "excess_kurtosis": 12.5,
                    "excess_kurtosis_rank": 3,
                },
                "fitting_statistics": {
                    "contrast": "logcosh",
                    "objective": 0.31,
                    "gaussian_objective": 0.37,
                    "absolute_deviation": 0.06,
                    "absolute_deviation_rank": 4,
                },
                "occurrences": [
                    {
                        "text": "�",
                        "token": "Ļ",
                        "token_id": 247,
                        "context": "interested in biology",
                        "score": -12.0,
                        "energy": 0.4,
                        "absolute_score_rank": 2,
                    }
                ],
                "logit_tokens": [{"text": "�", "token": "Ļ", "token_id": 247, "logit": 6.5}],
            }
        },
        logit_effects=(
            {
                "component": 0,
                "token_index": 0,
                "position": 3,
                "token_text": " hello",
                "multiplier": 1.1,
                "original_score": -2.0,
                "edited_score": -2.2,
                "tokens": [{"text": " world", "logit": 0.45}],
            },
        ),
    )


def test_write_explorer_html_is_self_contained_and_escapes_payload(tmp_path) -> None:
    destination = write_explorer_html(
        tmp_path / "nested" / "report.html",
        title="Test explorer",
        model="example/model@abc",
        layer=6,
        input_text="hello </script><script>alert(1)</script>",
        token_scope="assistant",
        messages=[
            {"role": "user", "content": "First turn"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second turn"},
            {"role": "assistant", "content": "Second response"},
        ],
        token_groups=[
            {
                "title": "Prompt template",
                "open": True,
                "tokens": [
                    {
                        "position": 0,
                        "token": "<|user|>",
                        "token_text": "<|user|>",
                        "top": [{"component": 7, "score": 2.5}],
                    }
                ],
            },
        ],
        tokens=[
            {
                "position": 3,
                "token": "Ġhello",
                "token_text": " hello",
                "top": [{"component": 2, "score": -1.25}],
            }
        ],
    )
    html = destination.read_text(encoding="utf-8")
    assert destination.is_file()
    assert "ICA Lens Explorer" in html
    assert "example/model@abc" in html
    assert "hello </script><script>" not in html
    assert "\\u003c/script>" in html
    assert 'selected ? color(component) : "#e7ebf1"' in html
    assert "Prompt template" in html
    assert "<|user|>" not in html
    assert "\\u003c|user|>" in html
    assert "details.open = Boolean(group.open)" in html
    assert "group.tokens.map(token => tokenCard(token, threshold))" in html
    assert '"metric":"score"' in html
    assert 'state.metric === "energy"' in html
    assert 'step="20" value="100"' in html
    assert 'value="3"' in html
    assert '<details class="panel analysis-context" id="analysisContext">' in html
    assert "<summary>Analysis details</summary>" in html
    assert 'document.getElementById("analysisContext").open = !data.compact' in html
    assert (
        'document.querySelector("main").append(document.getElementById("analysisContext"))' in html
    )
    assert 'class="token-position"' not in html
    assert 'content: "Analyzed text"' not in html
    assert "ui-monospace, SFMono-Regular" in html
    assert "white-space: nowrap; text-overflow: ellipsis" in html
    assert 'id="metricSelect"' in html
    assert 'id="topK"' in html
    assert "modelChip.title = data.model" in html
    assert '<span class="chip-label">Model:</span>' in html
    assert '<span class="chip-label">Token scope:</span>' in html
    assert '<span class="chip-label">Tokens:</span>' in html
    assert '"all text tokens": "entire input"' in html
    assert '<div class="context-label" id="inputLabel">Analyzed text</div>' in html
    assert '<blockquote class="input" id="input"></blockquote>' in html
    assert 'textContent = "Analyzed conversation"' in html
    assert "visible in ${visible}/${tokens.length} tokens" in html
    assert "main.scrollHeight * notebookScale" in html
    assert "const height = Math.max(contentHeight + 4, 180)" in html
    assert 'replace(/\\r\\n|\\r|\\n/g, "↵")' in html
    assert "max-width: 1680px" in html
    assert "`${data.result_group_title} (${data.tokens.length} tokens)`" in html
    assert '<details class="panel" id="resultGroup" open>' in html
    assert '"content":"Second turn"' in html
    assert '<span class="chip-label">User turns:</span>' in html
    assert "card.className = `message ${role}`" in html
    assert html.index('<section class="panel controls">') < html.index('<div id="tokenGroups">')


def test_analysis_iframe_scales_only_inside_vscode() -> None:
    rendered = analysis_iframe(analysis_result())

    assert "vscodeHost" in rendered
    assert "--vscode-font-family" in rendered
    assert "vscode-webview|vscode-resource" in rendered
    assert 0 < VSCODE_NOTEBOOK_SCALE <= 1
    assert f"document.documentElement.style.zoom = &quot;{VSCODE_NOTEBOOK_SCALE}&quot;" in rendered
    assert "main.scrollHeight * notebookScale" in rendered
    assert 'data-icalens-nested-frame="true"' in rendered
    assert "releaseWheelToNotebook" not in rendered
    assert "pointerEvents" not in rendered


def test_write_explorer_html_accepts_energy_metric(tmp_path) -> None:
    destination = write_explorer_html(
        tmp_path / "energy.html",
        title="Energy",
        model="example/model",
        layer=1,
        input_text="hello",
        token_scope="all",
        metric="energy",
        tokens=[],
    )

    assert '"metric":"energy"' in destination.read_text(encoding="utf-8")


def test_analysis_result_writes_html(tmp_path) -> None:
    result = analysis_result()

    destination = result.to_html(tmp_path / "analysis.html", metric="energy", top_k=1)
    html = destination.read_text(encoding="utf-8")
    assert '"model":"example/model@abc"' in html
    assert '"metric":"energy"' in html
    assert '"token_text":" hello"' in html
    assert '"token_label":" hello"' in html
    assert '"metrics":{"score":[-2.0,1.0]' in html
    assert '"top_k":1' in html
    assert '"dynamic_metrics":true' in html
    assert '"dominant_sign":"negative"' in html
    assert "Top occurrences:" in html
    assert "Top logit-lens tokens:" in html
    assert 'title="${esc(tooltip)}"' in html
    assert 'replace(/\\r\\n|\\r|\\n/g, "↵")' in html
    assert 'return lines.join("\\n")' in html
    assert '<details class="panel" id="componentProfile" hidden>' in html
    assert "Excess kurtosis" in html
    assert "Logcosh deviation" in html
    assert "|Logcosh contrast/objective − 0.374567|" in html
    assert "profile-score-statistics { display: flex; flex-direction: column; gap: 12px" in html
    assert ".profile-score-stat-value { color: #273244; font-size: 15px; }" in html
    assert ".profile-score-stat-rank { margin-left: 6px; color: #647084; font-size: 10px; }" in html
    assert "profile-score-stat-pair" not in html
    assert "grid-template-columns: minmax(0, 1fr) 150px" in html
    assert "Component profile — C${component} · ${direction}" in html
    assert "Tail selection" not in html
    assert "Skewness" in html
    profile_layout = html[html.index("body.innerHTML = `") :]
    assert profile_layout.index("Logit-lens tokens") < profile_layout.index("scoreStatistics")
    assert "Top-score occurrences · ${profile.dominant_sign}" in html
    assert "Logit-lens tokens · ${profile.dominant_sign}" in html
    assert "Suppressed logit-lens tokens" not in html
    assert 'class="profile-stat-row profile-stat-primary"' in html
    assert 'class="profile-chips"' in html
    assert "columns: 3 360px" in html
    assert 'class="profile-list profile-occurrences"' in html
    assert "Token ID:" in html
    assert "Raw token:" in html
    assert "Bytes:" in html
    assert "Show all" not in html
    assert "Show fewer" not in html
    assert "background: #f59e0b" in html
    assert "background: #1e3a8a" in html
    assert "lines.push(`#${index + 1} ${JSON.stringify(item.text)}`)" in html
    assert '<span class="profile-value">#${index + 1}</span>' in html
    assert 'profile-stat-row profile-stat-primary"><strong>Energy</strong>' in html
    assert 'profile-stat-row profile-stat-secondary"><strong>Positions</strong>' in html
    assert ".profile-stat-primary .profile-bar { height: 14px; }" in html
    assert 'title="Signed ICA score"' in html
    assert 'title="Component energy"' in html
    assert '<span class="profile-metric-icon">↕</span>' in html
    assert '<span class="profile-metric-icon">⚡</span>' in html
    assert 'class="profile-occurrence-token"' in html
    assert 'class="profile-occurrence-token" title="${esc(tokenHint(item))}"' in html
    assert 'class="profile-occurrence-metrics"' in html
    assert 'class="profile-occurrence-rank"' in html
    assert "Absolute-score rank among all components at this token" in html
    assert "(#${item.absolute_score_rank})" in html
    assert "font-size: 9px" in html
    assert "font-size: 13px; line-height: 1.5" in html
    assert "font-weight: inherit; text-decoration: underline" in html
    assert "profilePanel.hidden = true" in html
    assert "panel.hidden = false" in html
    assert 'class="profile-target"' in html
    assert "Number.isInteger(item.context_target_start)" in html
    assert "matches.reduce((best, index)" in html
    assert "highlightedContext(item)" in html
    assert 'data-token-index="${token.token_index}"' in html
    assert (
        "Local intervention projection — C${component} "
        "×${Number(effect.multiplier).toFixed(2)}" in html
    )
    assert "Ranked by absolute local logit change" in html
    assert "Number(item.logit).toFixed(3)" not in html
    assert '<details class="panel" id="localIntervention" hidden>' in html
    assert '"token_text":" hello","multiplier":1.1' in html


def test_gpt2_byte_hint_recovers_replacement_character_source_byte() -> None:
    assert _gpt2_byte_hex("Ļ") == "99"


def test_analysis_result_has_notebook_representation() -> None:
    rendered = analysis_result()._repr_html_()
    assert rendered.startswith('<iframe title="ICA Lens Explorer"')
    assert 'sandbox="allow-scripts allow-same-origin"' in rendered
    assert 'scrolling="no"' in rendered
    assert "overflow:hidden" in rendered
    assert "height:720px" in rendered
    assert "example/model@abc" in rendered
    assert "&lt;!doctype html&gt;" in rendered
    assert "&quot;compact&quot;:true" in rendered


def test_analysis_iframe_validates_options() -> None:
    result = analysis_result()
    rendered = analysis_iframe(result, metric="energy", top_k=1, height=400)
    assert "height:400px" in rendered
    assert "energy" in rendered


def test_analysis_iframe_initializes_selected_component() -> None:
    result = analysis_result()
    result = AnalysisResult(**{**result.__dict__, "selected_components": (1,)})

    rendered = analysis_iframe(result)

    assert "&quot;selected_components&quot;:[1]" in rendered
    assert "data.selected_components?.[0] ?? null" in rendered


def test_conversation_result_uses_message_token_groups() -> None:
    result = analysis_result()
    result = AnalysisResult(
        **{
            **result.__dict__,
            "messages": ({"role": "user", "content": "hello"},),
            "token_groups": ("User 1",),
        }
    )

    html = result._repr_html_()

    assert "User 1" in html
    assert "&quot;tokens&quot;:[]" in html
    assert "&quot;token_groups&quot;:[{&quot;title&quot;:&quot;User 1&quot;" in html

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from icalens._activation_dataset import (
    ActivationDataset,
    ActivationDatasetWriter,
    sample_metadata,
)


@dataclass
class _Document:
    input_ids: torch.Tensor


def test_activation_dataset_round_trip_and_resume(tmp_path) -> None:
    documents = [_Document(torch.tensor([10, 11, 12])), _Document(torch.tensor([20, 21]))]
    selected = {0: torch.tensor([0, 2]), 1: torch.tensor([1])}
    common = dict(
        model={"repo_id": "owner/model", "revision": "abc", "type": "base"},
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        layers=(0, 1),
        sample_count=3,
        hidden_size=2,
        dtype=torch.bfloat16,
        provenance={"dataset": {"repo_id": "owner/data"}},
        samples=sample_metadata(documents, selected),
    )
    writer = ActivationDatasetWriter(tmp_path, **common)
    layer_zero = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16)
    with writer.group((0,)) as sink:
        sink.append({0: layer_zero[:2]})
        sink.append({0: layer_zero[2:]})

    resumed = ActivationDatasetWriter(tmp_path, **common)
    assert resumed.missing_layers == (1,)
    layer_one = -layer_zero
    with resumed.group((1,)) as sink:
        sink.append({1: layer_one})
    resumed.finish()

    dataset = ActivationDataset(tmp_path)
    assert dataset.available_layers == (0, 1)
    torch.testing.assert_close(dataset.layer(0), layer_zero)
    torch.testing.assert_close(dataset.layer(1), layer_one)
    metadata = dataset.samples()
    assert metadata["document_index"].tolist() == [0, 0, 1]
    assert metadata["position"].tolist() == [0, 2, 1]
    assert metadata["token_id"].tolist() == [10, 12, 21]
    assert "manifest_sha256" in dataset.provenance["activation_dataset"]


def test_activation_dataset_rejects_incompatible_resume(tmp_path) -> None:
    arguments = dict(
        model={"repo_id": "owner/model", "revision": "abc", "type": "base"},
        activation_site="resid_post",
        layer_indexing="transformer_blocks_zero_based",
        layers=(0,),
        sample_count=1,
        hidden_size=2,
        dtype=torch.float32,
        provenance={},
        samples={"token_id": torch.tensor([1])},
    )
    ActivationDatasetWriter(tmp_path, **arguments)
    with pytest.raises(ValueError, match="incompatible"):
        ActivationDatasetWriter(tmp_path, **{**arguments, "hidden_size": 3})

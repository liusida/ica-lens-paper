from __future__ import annotations

import torch

from icalens._fastica import fit_fastica


def test_parallel_fastica_checkpoints_and_resume() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn((256, 4), generator=generator)
    checkpoints: dict[int, torch.Tensor] = {}

    full = fit_fastica(
        values,
        n_components=4,
        max_iter=4,
        random_state=3,
        row_normalize=False,
        checkpoint_callback=lambda iteration, unmixing, _whitening, _center: checkpoints.update(
            {iteration: unmixing.detach().clone()}
        ),
    )

    assert tuple(checkpoints) == (0, 1, 2, 3, 4)
    resumed_iterations: list[int] = []
    resumed = fit_fastica(
        values,
        n_components=4,
        max_iter=4,
        random_state=3,
        row_normalize=False,
        initial_unmixing=checkpoints[2],
        start_iteration=2,
        checkpoint_callback=lambda iteration, _unmixing, _whitening, _center: (
            resumed_iterations.append(iteration)
        ),
    )

    assert resumed_iterations == [3, 4]
    torch.testing.assert_close(resumed.components, full.components)
    torch.testing.assert_close(resumed.mixing, full.mixing)

    resumed_from_initial = fit_fastica(
        values,
        n_components=4,
        max_iter=4,
        random_state=999,
        row_normalize=False,
        initial_unmixing=checkpoints[0],
        start_iteration=0,
    )
    torch.testing.assert_close(resumed_from_initial.components, full.components)

from __future__ import annotations

from pathlib import Path
import inspect

import torch
import torch.nn as nn

from tracknet.evaluation.evaluator import _resolved_dataset_config
from tracknet.inference.aggregation import aggregate_window_outputs
from tracknet.inference.video_predictor import _sliding_window_indices
import tracknet.inference.video_predictor as video_predictor
from tracknet.models.registry import build_model
import tracknet.models.registry as model_registry
import tracknet.papers.v1.spec as v1_spec
import tracknet.papers.v2.spec as v2_spec
import tracknet.papers.v3.spec as v3_spec
import tracknet.papers.v4.spec as v4_spec
import tracknet.papers.v5.spec as v5_spec
from tracknet.data.dataset import ProcessedTrackNetDataset, TrackNetDatasetConfig
from tracknet.models.tracknet_v4 import TrackNetV4
from tracknet.models.tracknet_v5 import TrackNetV5
from tracknet.papers import get_paper_spec
from tracknet.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint


def test_paper_specs_expose_independent_dataset_contracts() -> None:
    v1 = get_paper_spec("v1")
    v3 = get_paper_spec("v3")
    v5 = get_paper_spec("v5")

    assert v1.dataset_defaults["heatmap_mode"] == "v1_uint8"
    assert v1.dataset_defaults["target_frame_mode"] == "last"
    assert v3.dataset_defaults["include_background"] is True
    assert v3.dataset_defaults["sequence_length"] == 8
    assert v5.dataset_defaults["heatmap_mode"] == "binary_disk"
    assert v5.dataset_defaults["sequence_length"] == 3
    assert get_paper_spec("v5_rstr").paper_id == "v5"


def test_paper_specs_are_owned_by_version_packages() -> None:
    assert v1_spec.build_spec().paper_id == "v1"
    assert v2_spec.build_spec().paper_id == "v2"
    assert v3_spec.build_tracker_spec().paper_id == "v3"
    assert v3_spec.build_rectifier_spec().paper_id == "v3_rectifier"
    assert v4_spec.build_spec().paper_id == "v4"
    assert v5_spec.build_spec().paper_id == "v5"


def test_v3_dataset_video_mixup_is_deterministic_and_mixes_targets(synthetic_processed_two_sequences_root: Path) -> None:
    cfg = TrackNetDatasetConfig(
        processed_root=synthetic_processed_two_sequences_root,
        sequence_length=3,
        target_frame_mode="all",
        heatmap_mode="binary_disk",
        radius=2.0,
        include_background=True,
        video_mixup_alpha=0.4,
        video_mixup_probability=1.0,
        seed=123,
    )
    ds_a = ProcessedTrackNetDataset(cfg)
    ds_b = ProcessedTrackNetDataset(cfg)

    mixed_a = ds_a[0]
    mixed_b = ds_b[0]
    clean = ProcessedTrackNetDataset(
        TrackNetDatasetConfig(
            processed_root=synthetic_processed_two_sequences_root,
            sequence_length=3,
            target_frame_mode="all",
            heatmap_mode="binary_disk",
            radius=2.0,
            include_background=True,
            seed=123,
        )
    )[0]

    assert mixed_a["mixup"]["enabled"] is True
    assert mixed_a["mixup"]["partner_index"] != 0
    assert torch.allclose(mixed_a["input"], mixed_b["input"])
    assert torch.allclose(mixed_a["target"], mixed_b["target"])
    assert not torch.allclose(mixed_a["target"], clean["target"])


class _RaisingOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("V4 must fuse high-level features before the heatmap output layer")


def test_v4_fuses_motion_before_heatmap_output_layer() -> None:
    model = TrackNetV4(sequence_length=3, base_channels=4, fusion_variant="eq4")
    model.backbone.output = _RaisingOutput()

    out = model(torch.rand(2, 9, 32, 32))

    assert out.shape == (2, 3, 32, 32)
    assert hasattr(model, "fusion_output")


def test_v5_rstr_uses_pixelshuffle_residual_decoder() -> None:
    model = TrackNetV5(base_channels=4, rstr_patch_size=8, rstr_embed_dim=16, rstr_heads=2, rstr_layers=1)

    assert any(isinstance(module, nn.PixelShuffle) for module in model.rstr.modules())
    assert model(torch.rand(1, 9, 32, 32)).shape == (1, 3, 32, 32)


def test_v5_registry_exposes_paper_ablation_models() -> None:
    for version in ["v5_mdd", "v5_rstr", "v5_full"]:
        model = build_model({"version": version, "base_channels": 4, "rstr_patch_size": 8, "rstr_embed_dim": 16, "rstr_heads": 2})
        assert model(torch.rand(1, 9, 32, 32)).shape == (1, 3, 32, 32)


def test_model_registry_accepts_paper_spec_aliases() -> None:
    assert build_model({"version": "tracknet_v1", "base_channels": 4})(torch.rand(1, 9, 32, 32)).shape == (1, 256, 32, 32)
    assert build_model({"version": "trajectory_rectifier", "hidden_channels": 4})(torch.rand(1, 4, 16)).shape == (1, 2, 16)


def test_model_registry_is_spec_driven_without_version_branch_table() -> None:
    source = inspect.getsource(model_registry.build_model)

    assert "if version in" not in source
    assert "Unknown model version" in source


def test_checkpoint_persists_amp_scaler_state(tmp_path: Path) -> None:
    model = nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "last.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        None,
        CheckpointState(epoch=0, global_step=3, best_score=0.5, metrics={"val_loss": 0.5}),
        {"model": {"version": "v2"}},
        scaler=scaler,
    )

    checkpoint = load_checkpoint(path)

    assert "scaler_state_dict" in checkpoint
    assert checkpoint["training"]["amp_scaler_saved"] is True


def test_evaluation_merges_paper_defaults_before_aggregation() -> None:
    dataset_cfg = _resolved_dataset_config({"processed_root": "data/processed"}, {"version": "v1"})

    assert dataset_cfg["target_frame_mode"] == "last"
    assert dataset_cfg["heatmap_mode"] == "v1_uint8"


def test_sliding_windows_cover_short_videos_without_dropping_frame_count() -> None:
    windows = _sliding_window_indices(n_frames=2, sequence_length=3)

    assert windows == [[0, 0, 0], [0, 0, 1]]


def test_video_prediction_implementation_does_not_materialize_all_window_inputs() -> None:
    source = inspect.getsource(video_predictor.run_video_prediction)

    assert "inputs = [" not in source


def test_aggregation_accepts_explicit_postprocess_kind_without_model_version_branch() -> None:
    outputs = torch.zeros(1, 1, 8, 8)
    outputs[0, 0, 3, 4] = 1.0

    predictions = aggregate_window_outputs(
        outputs,
        [[0]],
        postprocess_kind="largest_blob",
        sequence_length=1,
        target_frame_mode="all",
        threshold=0.5,
    )

    assert predictions[0].prediction.coordinate == (4.0, 3.0)


def test_aggregation_requires_explicit_postprocess_policy() -> None:
    outputs = torch.zeros(1, 1, 8, 8)

    try:
        aggregate_window_outputs(outputs, [[0]], sequence_length=1, target_frame_mode="all", threshold=0.5)
    except TypeError:
        return

    raise AssertionError("aggregation must be called with an explicit postprocess_kind")

from __future__ import annotations

from pathlib import Path
import inspect
import re

import torch
import torch.nn as nn

from tracknet.evaluation.evaluator import _resolved_dataset_config
from tracknet.inference.aggregation import aggregate_window_outputs
from tracknet.inference.video_predictor import _sliding_window_indices
import tracknet.inference.video_predictor as video_predictor
import tracknet.tools.predict_video as predict_video_tool
import tracknet.tools.visualize_dataset as visualize_dataset_tool
from tracknet.models.registry import build_model
import tracknet.models.registry as model_registry
import tracknet.evaluation.evaluator as evaluator
import tracknet.training.trainer as trainer
import tracknet.papers.v1.spec as v1_spec
import tracknet.papers.v2.spec as v2_spec
import tracknet.papers.v3.spec as v3_spec
import tracknet.papers.v4.spec as v4_spec
import tracknet.papers.v5.spec as v5_spec
from tracknet.data.dataset import ProcessedTrackNetDataset, TrackNetDatasetConfig
from tracknet.papers.base import HeatmapTargetPolicy
from tracknet.models.tracknet_v4 import TrackNetV4
from tracknet.models.tracknet_v1 import TrackNetV1
from tracknet.models.tracknet_v5 import DraftMDDFusion, MotionAwareDraftProjector, TrackNetV5
from tracknet.papers import get_paper_spec
from tracknet.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint


def test_paper_specs_expose_independent_dataset_contracts() -> None:
    v1 = get_paper_spec("v1")
    v3 = get_paper_spec("v3")
    v5 = get_paper_spec("v5")

    assert v1.target_policy is not None
    assert v1.target_policy.heatmap_mode == "v1_uint8"
    assert v1.target_policy.target_frame_mode == "last"
    assert v3.target_policy is not None
    assert v3.target_policy.include_background is True
    assert v3.dataset_defaults["sequence_length"] == 8
    assert v5.target_policy is not None
    assert v5.target_policy.heatmap_mode == "binary_disk"
    assert v5.dataset_defaults["sequence_length"] == 3
    assert get_paper_spec("v5_rstr").paper_id == "v5"


def test_pipeline_engines_do_not_embed_paper_version_branches() -> None:
    engine_sources = [
        inspect.getsource(trainer.TrackNetTrainer),
        inspect.getsource(evaluator.evaluate_checkpoint),
        inspect.getsource(video_predictor.run_video_prediction),
    ]

    forbidden_patterns = [
        r"['\"]v1['\"]",
        r"['\"]v2['\"]",
        r"['\"]v3['\"]",
        r"['\"]v4['\"]",
        r"['\"]v5['\"]",
        r"['\"]v3_rectifier['\"]",
        r"target_frame_mode",
        r"heatmap_mode",
    ]
    for source in engine_sources:
        lowered = source.lower()
        for pattern in forbidden_patterns:
            assert re.search(pattern, lowered) is None


def test_dataset_config_is_sampling_only_not_paper_semantics() -> None:
    signature = inspect.signature(TrackNetDatasetConfig)

    for field_name in ["model_version", "target_frame_mode", "heatmap_mode", "sigma", "radius", "include_background"]:
        assert field_name not in signature.parameters


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
    )
    policy = HeatmapTargetPolicy(
        target_frame_mode="all",
        heatmap_mode="binary_disk",
        radius=2.0,
        include_background=True,
        video_mixup_alpha=0.4,
        video_mixup_probability=1.0,
        seed=123,
    )
    ds_a = ProcessedTrackNetDataset(cfg, target_policy=policy)
    ds_b = ProcessedTrackNetDataset(cfg, target_policy=policy)

    mixed_a = ds_a[0]
    mixed_b = ds_b[0]
    clean = ProcessedTrackNetDataset(
        TrackNetDatasetConfig(
            processed_root=synthetic_processed_two_sequences_root,
            sequence_length=3,
        ),
        target_policy=HeatmapTargetPolicy(target_frame_mode="all", heatmap_mode="binary_disk", radius=2.0, include_background=True, seed=123),
    )[0]

    assert mixed_a["mixup"]["enabled"] is True
    assert mixed_a["mixup"]["partner_index"] != 0
    assert torch.allclose(mixed_a["input"], mixed_b["input"])
    assert torch.allclose(mixed_a["target"], mixed_b["target"])
    assert not torch.allclose(mixed_a["target"], clean["target"])


class _RaisingOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("V4 must fuse high-level features before the heatmap output layer")


class _RecordingFusionOutput(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, kernel_size=1)
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return super().forward(x)


def test_v4_fuses_motion_before_heatmap_output_layer() -> None:
    model = TrackNetV4(sequence_length=3, base_channels=4, fusion_variant="eq4")
    model.backbone.output = _RaisingOutput()
    recorder = _RecordingFusionOutput(12, 3)
    model.fusion_output = recorder

    out = model(torch.rand(2, 9, 32, 32))

    assert out.shape == (2, 3, 32, 32)
    assert hasattr(model, "fusion_output")
    assert recorder.last_input is not None
    assert recorder.last_input.shape[1] == 3 * 4


def test_v1_matches_paper_encoder_decoder_layer_depths() -> None:
    model = TrackNetV1(base_channels=4)

    assert model.encoder_depths == (4, 4, 8, 8, 16, 16, 16, 32, 32, 32)
    assert model.decoder_depths == (32, 32, 32, 8, 8, 4, 4, 256)
    assert model(torch.rand(1, 9, 32, 32)).shape == (1, 256, 32, 32)


def test_v5_rstr_uses_pixelshuffle_residual_decoder() -> None:
    model = TrackNetV5(base_channels=4, rstr_patch_size=8, rstr_embed_dim=16, rstr_heads=2, rstr_layers=1)

    assert any(isinstance(module, nn.PixelShuffle) for module in model.rstr.modules())
    assert model(torch.rand(1, 9, 32, 32)).shape == (1, 3, 32, 32)


def test_v5_draft_mdd_keeps_motion_maps_as_long_range_skip() -> None:
    fusion = DraftMDDFusion(sequence_length=3, motion_channels=4)
    draft = torch.rand(2, 3, 16, 16)
    motion = torch.rand(2, 4, 16, 16)

    fused = fusion(draft, motion)

    assert fused.shape == (2, 7, 16, 16)
    assert torch.allclose(fused[:, :3], draft)
    assert torch.allclose(fused[:, 3:], motion)


def test_v5_motion_aware_draft_projector_keeps_heatmap_base_three_channel() -> None:
    projector = MotionAwareDraftProjector(sequence_length=3, motion_channels=4)
    draft_mdd = torch.rand(2, 7, 16, 16)

    heatmap_base = projector(draft_mdd)

    assert heatmap_base.shape == (2, 3, 16, 16)


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
    assert "get_paper_spec" in source


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

    assert dataset_cfg["sequence_length"] == 3
    assert "target_frame_mode" not in dataset_cfg
    assert get_paper_spec("v1").target_policy is not None
    assert get_paper_spec("v1").target_policy.heatmap_mode == "v1_uint8"


def test_sliding_windows_cover_short_videos_without_dropping_frame_count() -> None:
    windows = _sliding_window_indices(n_frames=2, sequence_length=3)

    assert windows == [[0, 0, 0], [0, 0, 1]]


def test_video_prediction_implementation_does_not_materialize_all_window_inputs() -> None:
    source = inspect.getsource(video_predictor.run_video_prediction)

    assert "inputs = [" not in source
    assert "_read_video_frames" not in source
    assert "frames_bgr" not in source


def test_video_prediction_config_does_not_expose_paper_target_fields() -> None:
    signature = inspect.signature(video_predictor.VideoPredictionConfig)

    assert "target_frame_mode" not in signature.parameters
    assert "include_background" not in signature.parameters


def test_postprocess_requires_explicit_policy_not_model_version_guessing() -> None:
    import tracknet.inference.postprocess as postprocess

    signature = inspect.signature(postprocess.decode_model_output)

    assert "model_version" not in signature.parameters
    assert "default_postprocess_for_model" not in inspect.getsource(postprocess)


def test_cli_tools_do_not_parse_paper_target_fields() -> None:
    sources = [
        inspect.getsource(predict_video_tool.main),
        inspect.getsource(visualize_dataset_tool.main),
    ]

    for source in sources:
        assert "target_frame_mode" not in source
        assert "heatmap_mode" not in source
        assert "include_background" not in source


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

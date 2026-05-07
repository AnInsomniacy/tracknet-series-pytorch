"""Paper-level contracts for the TrackNet family."""

from __future__ import annotations

from tracknet.papers.base import PaperSpec
from tracknet.papers.v1.spec import build_spec as build_v1_spec
from tracknet.papers.v2.spec import build_spec as build_v2_spec
from tracknet.papers.v3.spec import build_rectifier_spec as build_v3_rectifier_spec
from tracknet.papers.v3.spec import build_tracker_spec as build_v3_spec
from tracknet.papers.v4.spec import build_spec as build_v4_spec
from tracknet.papers.v5.spec import build_spec as build_v5_spec


def _build_specs() -> dict[str, PaperSpec]:
    return {
        "v1": build_v1_spec(),
        "v2": build_v2_spec(),
        "v3": build_v3_spec(),
        "v3_rectifier": build_v3_rectifier_spec(),
        "v4": build_v4_spec(),
        "v5": build_v5_spec(),
    }


_SPECS: dict[str, PaperSpec] | None = None


def _specs() -> dict[str, PaperSpec]:
    global _SPECS
    if _SPECS is None:
        _SPECS = _build_specs()
    return _SPECS


def normalize_paper_id(name: str) -> str:
    key = name.lower().replace("tracknet_", "").replace("tracknet", "")
    aliases = {
        "rectifier": "v3_rectifier",
        "trajectory_rectifier": "v3_rectifier",
        "v5_mdd": "v5",
        "v5_rstr": "v5",
        "v5_full": "v5",
    }
    return aliases.get(key, key)


DEFAULT_PAPER_ID = "v2"


def paper_id_from_model_config(model_cfg: dict) -> str:
    return str(model_cfg.get("version", model_cfg.get("model_version", DEFAULT_PAPER_ID)))


def get_paper_spec(name: str) -> PaperSpec:
    key = normalize_paper_id(name)
    specs = _specs()
    if key not in specs:
        raise ValueError(f"Unknown TrackNet paper spec: {name}")
    return specs[key]


def dataset_config_with_paper_defaults(dataset_cfg: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return sampling defaults for compatibility with older callers.

    Paper-specific target generation now lives in `PaperSpec.target_policy`.
    This helper intentionally returns only neutral sampling configuration.
    """

    version = "v2"
    if model_cfg:
        version = str(model_cfg.get("version", model_cfg.get("model_version", version)))
    spec = get_paper_spec(version)
    return spec.resolved_dataset_config(dataset_cfg)

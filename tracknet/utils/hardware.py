"""Hardware capability reporting for reproducible experiment logs."""

from __future__ import annotations

import platform
from typing import Any

import cv2
import torch


def collect_hardware_report() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of available compute backends."""

    cuda_available = torch.cuda.is_available()
    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "cuda": {
            "available": bool(cuda_available),
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "devices": cuda_devices,
        },
        "mps": {
            "available": bool(mps_backend is not None and mps_backend.is_available()),
            "built": bool(mps_backend is not None and mps_backend.is_built()),
        },
    }

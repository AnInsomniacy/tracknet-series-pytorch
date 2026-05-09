"""Print local hardware capabilities for TrackNet pipeline runs."""

from __future__ import annotations

import json

from tracknet.utils.hardware import collect_hardware_report


def main() -> None:
    print(json.dumps(collect_hardware_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

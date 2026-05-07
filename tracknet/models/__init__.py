"""Model package exports."""

__all__ = ["build_model"]


def __getattr__(name: str):
    if name == "build_model":
        from tracknet.models.registry import build_model

        return build_model
    raise AttributeError(name)

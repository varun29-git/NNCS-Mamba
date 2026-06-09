"""Controller model interfaces and implementations."""

from nncs_mamba.models.controller import (
    ACTION_PRED_KEY,
    VALUE_PRED_KEY,
    ConstantController,
    ControllerCache,
    ControllerProtocol,
    ControllerSpec,
)

_optional_model_exports = []

try:
    from nncs_mamba.models.heads import ActionHead, MLPHead, ValueHead
    from nncs_mamba.models.mamba import (
        DualHeadMambaController,
        MambaControllerConfig,
        SelectiveSSMBlock,
    )

    _optional_model_exports = [
        "ActionHead",
        "DualHeadMambaController",
        "MLPHead",
        "MambaControllerConfig",
        "SelectiveSSMBlock",
        "ValueHead",
    ]
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise


__all__ = [
    "ACTION_PRED_KEY",
    "VALUE_PRED_KEY",
    "ConstantController",
    "ControllerCache",
    "ControllerProtocol",
    "ControllerSpec",
] + _optional_model_exports

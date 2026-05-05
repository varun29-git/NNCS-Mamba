from typing import Any, Dict

from gru_learner import GRUController
from mlp_learner import MLPController
from mamba_learner import MambaController


def build_controller(
    controller_type: str = "mamba",
    *,
    obs_dim: int,
    action_dim: int,
    d_model: int = 128,
    d_state: int = 16,
    num_layers: int = 3,
    lr: float = 3e-4,
    optimizer_name: str = "split_muon",
    use_gradient_checkpointing: bool = True,
    aux_state_weight: float = 0.5,
    late_timestep_weight: float = 1.0,
    recurrent_dropout: float = 0.1,
    action_clip=None,
):
    controller_key = (controller_type or "mamba").lower()
    if controller_key == "mamba":
        return MambaController(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            lr=lr,
            optimizer_name=optimizer_name,
            use_gradient_checkpointing=use_gradient_checkpointing,
            aux_state_weight=aux_state_weight,
            late_timestep_weight=late_timestep_weight,
            action_clip=action_clip,
        )
    if controller_key == "gru":
        return GRUController(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            lr=lr,
            optimizer_name="adamw",
            use_gradient_checkpointing=False,
            aux_state_weight=aux_state_weight,
            late_timestep_weight=late_timestep_weight,
            recurrent_dropout=recurrent_dropout,
            action_clip=action_clip,
        )
    if controller_key == "mlp":
        return MLPController(
            obs_dim=obs_dim,
            action_dim=action_dim,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            lr=lr,
            optimizer_name="adamw",
            use_gradient_checkpointing=False,
            aux_state_weight=aux_state_weight,
            late_timestep_weight=late_timestep_weight,
            recurrent_dropout=recurrent_dropout,
            action_clip=action_clip,
        )
    raise ValueError(f"Unsupported controller_type: {controller_type}")


def build_controller_from_config(config: Dict[str, Any]):
    return build_controller(
        controller_type=config.get("controller_type", "mamba"),
        obs_dim=config.get("obs_dim", 15),
        action_dim=config.get("action_dim", 4),
        d_model=config.get("d_model", 128),
        d_state=config.get("d_state", 16),
        num_layers=config.get("num_layers", 3),
        lr=config.get("lr", 3e-4),
        optimizer_name=config.get("optimizer_name", "split_muon"),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", True),
        aux_state_weight=config.get("aux_state_weight", 0.5),
        late_timestep_weight=config.get("late_timestep_weight", 1.0),
        recurrent_dropout=config.get("recurrent_dropout", 0.1),
        action_clip=config.get("action_clip"),
    )

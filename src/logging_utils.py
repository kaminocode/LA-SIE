from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    import wandb
except ImportError:
    wandb = None


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(val) for val in value]
    return value


def args_to_serializable_dict(args: Any) -> Dict[str, Any]:
    return {key: _to_serializable(value) for key, value in vars(args).items()}


class ScalarLogger:
    def __init__(
        self,
        writer=None,
        *,
        use_wandb: bool = False,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        run_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.writer = writer
        self._wandb_run = None
        self._wandb_buffer: Dict[str, Any] = {}
        self._wandb_step: Optional[int] = None

        if not use_wandb:
            return

        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it or run without --wandb."
            )

        init_kwargs = {
            "config": _to_serializable(config or {}),
            "name": name,
        }
        if project:
            init_kwargs["project"] = project
        if entity:
            init_kwargs["entity"] = entity
        if run_dir is not None:
            init_kwargs["dir"] = str(run_dir)
        self._wandb_run = wandb.init(**init_kwargs)
        if self._wandb_run is not None:
            self._wandb_run.define_metric("global_step")
            self._wandb_run.define_metric("*", step_metric="global_step")

    def add_scalar(self, name: str, value: Any, step: int) -> None:
        scalar_value = _to_serializable(value)
        if self.writer is not None:
            self.writer.add_scalar(name, scalar_value, step)
        if self._wandb_run is None:
            return
        if self._wandb_step is None:
            self._wandb_step = step
        elif step != self._wandb_step:
            self._flush_wandb()
            self._wandb_step = step
        self._wandb_buffer[name] = scalar_value

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()
        self._flush_wandb()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _flush_wandb(self) -> None:
        if self._wandb_run is None or not self._wandb_buffer:
            return
        payload = dict(self._wandb_buffer)
        if self._wandb_step is not None:
            payload["global_step"] = self._wandb_step
        self._wandb_run.log(payload)
        self._wandb_buffer.clear()

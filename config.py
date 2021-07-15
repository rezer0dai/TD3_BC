from dataclasses import dataclass, fields
from typing import Any

@dataclass
class DefaultVal:
    val: Any

@dataclass
class NoneRefersDefault:
    def __post_init__(self):
        for field in fields(self):
            if isinstance(field.default, DefaultVal):
                field_val = getattr(self, field.name)
                if isinstance(field_val, DefaultVal) or field_val is None:
                    setattr(self, field.name, field.default.val)

@dataclass
class Config(NoneRefersDefault):
    env: str = "panda-reach"
    seed: int = 0
    eval_freq: int = 5e3
    max_timesteps: int = 1e6
    # TD3
    expl_noise: float = 0.1
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    # TD3 + BC
    alpha: float = 2.5
    normalize: bool = True
    # OPEN AI TD3 BASELINE TRAINING
    steps_per_epoch: int = 4000
    epochs: int =100
    replay_size: int = 1e6
    start_steps: int = 10000
    update_after: int = 1000
    update_every: int = 50
    her_per_ep: int = 20


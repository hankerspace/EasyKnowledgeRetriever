from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable
from .base import BaseConfig

@dataclass
class RerankConfig(BaseConfig):
    model_func: Optional[Callable[..., object]] = None
    # Add other rerank specific configs as needed, currently EKR uses generic params

from dataclasses import dataclass, field


@dataclass
class SessionState:
    provider: str = ""
    model: str = ""
    aspect_ratio: str = ""
    res_key: str = ""
    quality: str = ""
    cost: float = 0.0
    prompt: str = ""
    original_prompt: str = ""
    dual: bool = False
    images: list[str] = field(default_factory=list)

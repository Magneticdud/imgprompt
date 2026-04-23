from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationRequest:
    prompt: str
    model: str
    aspect_ratio: str
    res_key: str
    quality_key: str
    images: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    input_pixels: int = 0
    width: int | None = None
    height: int | None = None
    input_pixels: int = 0

    @property
    def is_batch(self) -> bool:
        return len(self.images) > 1

    @property
    def is_text_to_image(self) -> bool:
        return len(self.images) == 0

    @property
    def primary_image(self) -> str | None:
        return self.images[0] if self.images else None


class ImageProvider(ABC):
    @abstractmethod
    def run(self, request: GenerationRequest) -> None:
        """Execute the generation/edit request and save result(s) to disk."""
        ...

    @property
    def supports_batch(self) -> bool:
        return True

    @property
    def supports_dual(self) -> bool:
        return False

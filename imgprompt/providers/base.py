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
    # Number of variants to generate per API call (1..10). Defaults to 1 so
    # replay/history files written before this field was added still load,
    # since dataclass __init__ allows **unpacking without the key when a
    # default is provided.
    n: int = 1
    # Free-form passthrough for advanced parameters not yet modelled in the
    # provider interface (output_format, background, seed, provider.options,
    # etc.). Forwarded verbatim into the request body for providers that
    # support it.
    extras: dict = field(default_factory=dict)

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

    @classmethod
    @abstractmethod
    def provider_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def supported_models(cls) -> list[str]: ...

    @classmethod
    def default_model(cls) -> str:
        return cls.supported_models()[0]

    @abstractmethod
    def get_resolution_choices(
        self, model: str, image_path: str | None
    ) -> tuple[list[str], str]: ...

    @abstractmethod
    def resolve_resolution(
        self, model: str, selection: str
    ) -> tuple[str, int | None, int | None]: ...

    @abstractmethod
    def get_quality_choices(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        image_path: str | None,
    ) -> tuple[list[str], str]: ...

    @abstractmethod
    def resolve_quality(
        self,
        model: str,
        res_key: str,
        width: int | None,
        height: int | None,
        selection: str,
    ) -> tuple[str, float]: ...

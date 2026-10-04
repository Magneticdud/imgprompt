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
    # Dual-input intent: both images belong to ONE api call (IMG_1 + IMG_2
    # referenced together in the prompt), as opposed to batch mode where each
    # image is an independent single-input call. Defaults to False so
    # .last_generation.json files saved before this field existed still load.
    is_dual: bool = False
    # Generalized combined-input intent: 3+ images all reach the model in ONE
    # prompt/call (IMG_1..IMG_N referenced together), as opposed to batch mode
    # where each image is processed independently. is_dual is the 2-image
    # special case kept for schema compatibility; together they form
    # `is_combined` below. Defaults to False so older history files load.
    is_multi: bool = False
    # Wizard's pre-call USD estimate for one API call. Used to reconcile
    # against the provider-reported usage.cost after the call (a >10%
    # divergence is printed). None when no estimate was shown (non-wizard
    # entry points, providers without cost preview).
    estimated_cost: float | None = None

    @property
    def is_combined(self) -> bool:
        # Any multi-image selection that must reach the provider in a single
        # call: dual (exactly 2) or multi (3+).
        return self.is_dual or self.is_multi

    @property
    def is_batch(self) -> bool:
        # Combined mode also carries 2+ images but is NOT a batch: they must
        # reach the provider in a single call, not one call each.
        return len(self.images) > 1 and not self.is_combined

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

    def max_input_images(self, model: str) -> int | None:
        """Maximum number of input/reference images this model accepts in one
        combined call, or None when the provider advertises no numeric bound.

        The wizard prints this next to a combined-mode summary so the user
        knows before spending whether some images will be trimmed. Providers
        with live capability descriptors (OpenRouter) override this to read
        the descriptor and fall back to their hardcoded table.
        """
        return None

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

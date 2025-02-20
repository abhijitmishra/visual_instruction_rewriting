# Reference : https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/ReVision/__init__.py

from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {"configuration_revision": ["ReVisionConfig"]}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_revision"] = [
        "ReVisionForConditionalGeneration",
        "ReVisionPreTrainedModel",
    ]
    _import_structure["processing_revision"] = ["ReVisionProcessor"]


if TYPE_CHECKING:
    from .configuration_revision import ReVisionConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_revision import (
            ReVisionForConditionalGeneration,
            ReVisionPreTrainedModel,
        )
        from .processing_revision import ReVisionProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure
    )

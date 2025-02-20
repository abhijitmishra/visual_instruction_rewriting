# Reference: https://github.com/huggingface/transformers/tree/main/src/transformers/models/paligemma
"""ReVisionmodel configuration"""

import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


class ReVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ReVisionForConditionalGeneration`]. It is used to instantiate an
    ReVisionmodel according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ReVision-500M.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`ReVisionVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `MistralConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 256000):
            The image token index to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size of the ReVisionmodel. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ReVisionForConditionalGeneration`]
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimension of the multimodal projection space.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden layer of the Language model.

    Example:

    ```python
    >>> from transformers import ReVisionForConditionalGeneration, ReVisionConfig, SigLipVisionConfig, MistralConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SigLipVisionConfig()

    >>> # Initializing a ReVision config
    >>> text_config = MistralConfig()

    >>> # Initializing a ReVision revision-500M style configuration
    >>> configuration = ReVisionConfig(vision_config, text_config)

    >>> # Initializing a model from the ReVision revision-500M style configuration
    >>> model = ReVisionForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "revision"
    is_composition = False

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        vocab_size=32768,
        projection_dim=768,
        hidden_size=768,
        sampler_tokens=64,
        **kwargs,
    ):
        self._ignore_index = ignore_index
        self.image_token_index = image_token_index
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.sampler_tokens = sampler_tokens
        self.is_encoder_decoder = False

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"]
                if "model_type" in vision_config
                else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](
                **vision_config
            )
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=3072,
                hidden_size=768,
                patch_size=16,
                image_size=384,
                num_hidden_layers=12,
                num_attention_heads=12,
                vocab_size=32000,
                vision_use_head=False,
            )

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = (
                text_config["model_type"] if "model_type" in text_config else "mistral"
            )
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["mistral"](
                hidden_size=768,
                num_hidden_layers=12,
                intermediate_size=3072,
                num_attention_heads=16,
                num_key_value_heads=8,
                is_encoder_decoder=False,
                vocab_size=vocab_size,
            )
        self.text_config.num_image_tokens = int(
            (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        )
        self.vision_config.projection_dim = projection_dim
        super().__init__(**kwargs)

    @property
    def ignore_index(self):
        warnings.warn(
            "The `ignore_index` attribute is deprecated and will be removed in v4.47.",
            FutureWarning,
        )
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._ignore_index = value

    @property
    def vocab_size(self):
        warnings.warn(
            "The `vocab_size` attribute is deprecated and will be removed in v4.44, Please use `text_config.vocab_size` instead.",
            FutureWarning,
        )
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        output.pop("_ignore_index", None)
        return output

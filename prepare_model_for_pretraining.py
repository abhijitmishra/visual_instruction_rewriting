# We load lightweight already pretrained Vision and Text models
# Right now we support CLIP and Mistral
# sample command:
# Example1: python prepare_model_for_pretraining.py --vision_model_name_or_path google/siglip-base-patch16-256 --text_model_name_or_path Qwen/Qwen2-0.5B-Instruct --dest ../ReVision-587M-256-16-random/ --text_model_type qwen
# python prepare_model_for_pretraining.py --vision_model_name_or_path google/siglip-base-patch16-256 --text_model_name_or_path OuteAI/Lite-Mistral-150M-v2-Instruct --dest ../ReVision-250M-64-16-random

# # Number of parameters: 171176960

from transformers import (
    SiglipVisionModel,
    MistralForCausalLM,
    GPTNeoForCausalLM,
    LlamaTokenizer,
    GPT2Tokenizer,
    SiglipImageProcessor,
    MistralForCausalLM,
    LlamaTokenizer,
)
from model import (
    ReVisionConfig,
    ReVisionForConditionalGeneration,
    ReVisionProcessor,
)

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Prepare model for pretraining.")

    # Required arguments
    parser.add_argument(
        "--vision_model_name_or_path",
        type=str,
        required=True,
        help="Path to the vision model or its name in the Hugging Face model hub.",
    )
    parser.add_argument(
        "--text_model_name_or_path",
        type=str,
        required=True,
        help="Path to the text model or its name in the Hugging Face model hub.",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Destination path where the prepared model will be saved.",
    )

    # Optional arguments
    parser.add_argument(
        "--vision_model_type",
        type=str,
        default="siglip",
        help="Type of vision model (default: 'clip').",
    )
    parser.add_argument(
        "--text_model_type",
        type=str,
        default="mistral",
        help="Type of text model (default: 'mistral').",
    )

    return parser.parse_args()


def upload_to_hf(model_path, **kwargs):
    return None


def prepare_model_for_pretraining(
    vision_model_name_or_path: str,
    text_model_name_or_path: str,
    dest: str,
    vision_model_type="siglip",
    text_model_type="mistral",
):
    if "siglip" not in vision_model_type or "mistral" not in text_model_type:
        raise NotImplementedError("Only SigLIP and Mistral backbones are supported")
    vision_model = SiglipVisionModel.from_pretrained(vision_model_name_or_path)

    text_model = MistralForCausalLM.from_pretrained(text_model_name_or_path)
    tokenizer = LlamaTokenizer.from_pretrained(text_model_name_or_path)

    vision_model_config = vision_model.config
    text_model_config = text_model.config

    config = ReVisionConfig(
        vision_config=vision_model_config, text_config=text_model_config
    )
    vision_processor = SiglipImageProcessor.from_pretrained(vision_model_name_or_path)

    processor = ReVisionProcessor(image_processor=vision_processor, tokenizer=tokenizer)
    model = ReVisionForConditionalGeneration(config=config)
    model.vision_tower = vision_model
    model.language_model = text_model
    model.save_pretrained(dest)
    processor.save_pretrained(dest)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")


if __name__ == "__main__":
    args = get_args()
    prepare_model_for_pretraining(
        vision_model_name_or_path=args.vision_model_name_or_path,
        text_model_name_or_path=args.text_model_name_or_path,
        dest=args.dest,
        vision_model_type=args.vision_model_type,
        text_model_type=args.text_model_type,
    )

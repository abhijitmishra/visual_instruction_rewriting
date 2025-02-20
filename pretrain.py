from model import ReVisionProcessor, ReVisionForConditionalGeneration
from datautils import LLAVADataset, LLAVARecapDataset, LLAVADatasetCC3M, CombinedDataset
from args import get_args_pretraining
from transformers import (
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
import os
import random


os.environ["HF_DATASETS_CACHE"] = "/media/anon/data/blabla/cache/"  # "/root/cache"
MODEL_ID = "./ReVision-250M-64-16-random"

# vision_model_name_or_path = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
vision_model_name_or_path = "google/siglip-base-patch16-256"
text_model_name_or_path = "OuteAI/Lite-Mistral-150M-v2-Instruct"


def set_seed(seed):
    """Set seed for reproducibility"""
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable to ensure reproducibility


def main(args):
    use_auth_token = os.getenv("HF_TOKEN")

    model = ReVisionForConditionalGeneration.from_pretrained(
        MODEL_ID, use_auth_token=use_auth_token
    )
    processor = ReVisionProcessor.from_pretrained(MODEL_ID)
    dataset1 = LLAVADataset(processor=processor)
    dataset2 = LLAVARecapDataset()
    dataset3 = LLAVADatasetCC3M()
    dataset = CombinedDataset(datasets=[dataset1, dataset2, dataset3])

    # If you want to freeze the vision and text towers, enable this:
    # Ideal for pretraining
    # for param in model.vision_tower.parameters():
    #     param.requires_grad = False
    # for param in model.language_model.parameters():
    #     param.requires_grad = False

    trainer_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=args.remove_unused_columns,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        logging_steps=args.logging_steps,
        optim=args.optim,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        output_dir=args.output_dir,
        gradient_checkpointing=False,
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=dataset1.collate_fn,
        args=trainer_args,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Pushing to hub")
    model.push_to_hub(
        "anonymoususerrevision/ReVision-250M-64-16", use_auth_token=use_auth_token
    )
    processor.push_to_hub(
        "anonymoususerrevision/ReVision-250M-64-16", use_auth_token=use_auth_token
    )

    # print("training complete")


if __name__ == "__main__":

    args = get_args_pretraining()
    main(args)

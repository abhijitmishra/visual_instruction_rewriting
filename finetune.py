from model import ReVisionProcessor, ReVisionForConditionalGeneration
from datautils import RevisionRewriteDataset, SyntheticRewriteDataset, CombinedDataset
from args import get_args_fine_tuning
from transformers import (
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
import os
import random
from PIL import Image

# MODEL_ID = "anonymoususerrevision/ReVision-171M-224px-random"
MODEL_ID = "anonymoususerrevision/ReVision-250M-256-16"


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


# Define a custom Trainer class
class CustomTrainer(Trainer):
    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0  # Initialize step counter
        self.processor = processor

    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        # Generate samples every 100 steps
        if self.step_count % 100 == 0:
            self.generate_samples()

    def generate_samples(self):
        # Define a few input prompts for sample generation
        image = Image.open(
            "/home/anon/Desktop/img.jpg"
        )  # requests.get(url, stream=True).raw)
        image = image.convert("RGB")
        # default_size = (100, 100)
        # default_color = (255, 255, 255)  # White color
        # image = Image.new("RGB", default_size, default_color)

        # Instruct the model to create a caption in Spanish
        prompt = "rewrite: who wrote this book?"
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            print(decoded)


def main(args):
    set_seed(42)
    use_auth_token = os.getenv("HF_TOKEN")

    model = ReVisionForConditionalGeneration.from_pretrained(
        MODEL_ID, use_auth_token=use_auth_token
    )

    processor = ReVisionProcessor.from_pretrained(
        MODEL_ID, use_auth_token=use_auth_token
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dataset1 = RevisionRewriteDataset(
        split="train", use_auth_token=use_auth_token, processor=processor
    )

    for param in model.vision_tower.parameters():
        param.requires_grad = False

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
        bf16=False,
        gradient_checkpointing=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )

    trainer = CustomTrainer(
        processor=processor,
        model=model,
        train_dataset=dataset1,
        data_collator=dataset1.collate_fn,
        args=trainer_args,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Pushing to hub")
    model.push_to_hub(
        "anonymoususerrevision/ReVision-250M-256-16-baseline", use_auth_token=use_auth_token
    )
    processor.push_to_hub(
        "anonymoususerrevision/ReVision-250M-256-16-baseline", use_auth_token=use_auth_token
    )

    # print("training complete")


if __name__ == "__main__":

    args = get_args_fine_tuning()
    main(args)

from model import ReVisionProcessor, ReVisionForConditionalGeneration
# from datautils import RevisionRewriteDatasetWithMetadata
from args import get_args_fine_tuning
from transformers import (
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch
import os
import random
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import zipfile
from huggingface_hub import cached_assets_path, hf_hub_download
import pandas as pd
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




class RevisionRewriteDatasetWithSelfMetadata(Dataset):
    def __init__(
        self,
        dataset_name="anonymoususerrevision/multimodal_query_rewrites",
        use_auth_token=None,
        processor=None,
        split="train",
    ):
        # self.dataset = load_dataset(dataset_name)
        cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
        self.processor = processor
        if cache_dir is None:
            self.image_zip_path = hf_hub_download(
                repo_id=dataset_name,
                filename="images.zip",
                repo_type="dataset",
                use_auth_token=use_auth_token,
            )
            self.dataset_path = hf_hub_download(
                repo_id=dataset_name,
                filename="train_with_metadata.tsv" if split == "train" else "test_with_metadata.tsv",
                repo_type="dataset",
                use_auth_token=use_auth_token,
            )
        else:
            self.image_zip_path = hf_hub_download(
                repo_id=dataset_name,
                filename="images.zip",
                repo_type="dataset",
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
            )
            self.dataset_path = hf_hub_download(
                repo_id=dataset_name,
                filename="train_with_metadata.tsv" if split == "train" else "test_with_metadata.tsv",
                repo_type="dataset",
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
            )

        # with open(self.dataset_path) as f:
        #     self.dataset = pd.read_csv(self.dataset_path, sep="\t")
        #     print(f"length of data {len(self.dataset)}")

        # self.dataset = pd.read_csv("train_selfmetadata.tsv", sep="\t")
        self.dataset = pd.read_csv("train_tempthing.tsv", sep="\t")
        self.image_extract_path = os.path.join(
            os.path.dirname(self.image_zip_path), "images/images"
        )

        if not os.path.exists(self.image_extract_path):
            print(f"Extracting image to {self.image_extract_path}")
            with zipfile.ZipFile(self.image_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.image_extract_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the data entry
        data = self.dataset.iloc[idx]

        # # Process image file path
        image_file = data["Image Id"] + ".jpg"
        image_path = os.path.join(self.image_extract_path, image_file)
        image = Image.open(image_path).convert("RGB")

        # Process prompt and response from conversation
        prompt_with_metadata = data["PromptWithMetadata"]
        response = data["Rewritten Question"]
        # caption = data["Captions"]

        # ocr_text = str(data["OCRText"])

        # Append Prompt with "<task>" tag
        # prompt = "<task> " + prompt

        # data_section = "<data> " + caption
        # if len(ocr_text) > 0:
        #     data_section += " The text in the image is: " + ocr_text
        
        # prompt_with_metadata = prompt + data_section

        return image, prompt_with_metadata, response
        # return data["Image Id"], data["Prompt"], data["Rewritten Question"], data["OCRText"]

    # Define the collate function
    def collate_fn(self, examples, to_bf16=True):
        # Separate images, prompts, and responses

        texts = [example[1].replace("\n", "") for example in examples]  # Prompt
        labels = [example[2].replace("\n", "") for example in examples]  # Response
        images = [
            example[0].convert("RGB") for example in examples
        ]  # Convert images to RGB

        tokens = self.processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
            tokenize_newline_separately=False,
        )

        if to_bf16:
            tokens = tokens.to(torch.bfloat16)
        return tokens



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

    dataset = RevisionRewriteDatasetWithSelfMetadata(
        split="train", use_auth_token=use_auth_token, processor=processor
    )

    # If you want to freeze the vision and text towers, enable this:
    # Ideal for pretraining
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
        train_dataset=dataset,
        data_collator=dataset.collate_fn,
        args=trainer_args,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Pushing to hub")
    model.push_to_hub(
        "anonymoususerrevision/ReVision-250M-256-16-selfmetadata",
        use_auth_token=use_auth_token,
        private=True,
    )
    processor.push_to_hub(
        "anonymoususerrevision/ReVision-250M-256-16-selfmetadata",
        use_auth_token=use_auth_token,
        private=True,
    )

    # print("training complete")


if __name__ == "__main__":
    args = get_args_fine_tuning()
    main(args)

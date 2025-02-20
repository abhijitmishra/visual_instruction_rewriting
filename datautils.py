from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageDraw
from huggingface_hub import cached_assets_path, hf_hub_download

import json
import torch
import os
import base64
import re
import zipfile
import pandas as pd

MAX_LEN = 512


class LLAVARecapDataset(Dataset):
    def __init__(self) -> None:
        cache_dir = os.environ["HF_DATASETS_CACHE"]
        if cache_dir is not None:
            print(cache_dir)
            self.dataset = load_dataset(
                path="lmms-lab/LLaVA-ReCap-CC3M", split="train", cache_dir=cache_dir
            )
        else:
            self.dataset = load_dataset(
                path="lmms-lab/LLaVA-ReCap-CC3M",
                split="train",
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # Decode the image from Base64

        image = data["image"]
        prompt = "what is this?"
        try:
            response = data["conversations"][1]["value"]
        except:
            response = None
        # image = image.resize((224, 224))
        if image is None:
            print("Training data issue")
            default_size = (256, 256)
            default_color = (255, 255, 255)  # White color
            image = Image.new("RGB", default_size, default_color)

            # Optional: Draw something on the default image
            draw = ImageDraw.Draw(image)
            draw.text((10, 10), "Default Image", fill=(0, 0, 0))  # Add some text
            prompt = "No prompt"
            response = "No response"
        if prompt is None or response is None:
            print("Training data issue")
            prompt = "No prompt"
            response = "No response"
        return image, prompt, response


class LLAVADataset(Dataset):
    def __init__(self, dataset_name="liuhaotian/LLaVA-Pretrain", processor=None):
        # self.dataset = load_dataset(dataset_name)
        cache_dir = os.environ["HF_DATASETS_CACHE"]

        self.image_zip_path = hf_hub_download(
            repo_id=dataset_name,
            filename="images.zip",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        self.dataset_path = hf_hub_download(
            repo_id=dataset_name,
            filename="blip_laion_cc_sbu_558k.json",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        self.processor = processor

        with open(self.dataset_path) as f:
            self.dataset = json.load(f)
            print(f"length of data {len(self.dataset)}")

        self.image_extract_path = os.path.join(
            os.path.dirname(self.image_zip_path), "images"
        )

        if not os.path.exists(self.image_extract_path):
            print(f"Extracting image to {self.image_extract_path}")
            with zipfile.ZipFile(self.image_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.image_extract_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the data entry
        data = self.dataset[idx]

        # Process image file path
        image_file = data["image"]
        image_path = os.path.join(self.image_extract_path, image_file)
        image = Image.open(image_path).convert("RGB")

        # Process prompt and response from conversation
        conversation = data["conversations"]
        prompt = (
            "what is this?"  # conversation[0]["value"].replace("<image>", "").strip()
        )
        response = conversation[1]["value"]
        return image, prompt, response

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
            max_length=MAX_LEN,
            padding="longest",
            truncation=True,
            tokenize_newline_separately=False,
        )
        if to_bf16:
            tokens = tokens.to(torch.bfloat16)
        return tokens


class LLAVADatasetCC3M(Dataset):
    def __init__(
        self, dataset_name="liuhaotian/LLaVA-CC3M-Pretrain-595K", processor=None
    ):
        # self.dataset = load_dataset(dataset_name)
        cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
        if cache_dir is not None:
            self.image_zip_path = hf_hub_download(
                repo_id=dataset_name,
                filename="images.zip",
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            self.dataset_path = hf_hub_download(
                repo_id=dataset_name,
                filename="chat.json",
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        else:
            self.image_zip_path = hf_hub_download(
                repo_id=dataset_name,
                filename="images.zip",
                repo_type="dataset",
            )
            self.dataset_path = hf_hub_download(
                repo_id=dataset_name,
                filename="chat.json",
                repo_type="dataset",
            )
        self.processor = processor

        with open(self.dataset_path) as f:
            self.dataset = json.load(f)
            print(f"length of data {len(self.dataset)}")

        self.image_extract_path = os.path.join(
            os.path.dirname(self.image_zip_path), "images"
        )

        if not os.path.exists(self.image_extract_path):
            print(f"Extracting image to {self.image_extract_path}")
            with zipfile.ZipFile(self.image_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.image_extract_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the data entry
        data = self.dataset[idx]

        # Process image file path
        image_file = data["image"]
        image_path = os.path.join(self.image_extract_path, image_file)
        image = Image.open(image_path).convert("RGB")

        # Process prompt and response from conversation
        conversation = data["conversations"]
        prompt = "what is this?"
        # prompt = conversation[0]["value"].replace("<image>", "").strip()
        response = conversation[1]["value"]
        # print(f"Prompt {prompt}")
        # print(f"res {response}")
        return image, prompt, response


class RevisionRewriteDataset(Dataset):
    def __init__(
        self,
        dataset_name="anonymoususerrevision/multimodal_query_rewrites",
        use_auth_token=None,
        processor=None,
        split="train",
        add_image_prefix=False,
        return_file_name=False,
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
                filename="train.tsv" if split == "train" else "test.tsv",
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
                filename="train.tsv" if split == "train" else "test.tsv",
                repo_type="dataset",
                cache_dir=cache_dir,
                use_auth_token=use_auth_token,
            )

        with open(self.dataset_path) as f:
            self.dataset = pd.read_csv(self.dataset_path, sep="\t")
            print(f"length of data {len(self.dataset)}")

        self.image_extract_path = os.path.join(
            os.path.dirname(self.image_zip_path), "images"
        )

        if not os.path.exists(self.image_extract_path):
            print(f"Extracting image to {self.image_extract_path}")
            with zipfile.ZipFile(self.image_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.image_extract_path)
        self.add_image_prefix = add_image_prefix
        self.return_file_name = return_file_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the data entry
        data = self.dataset.iloc[idx]

        # Process image file path
        image_file = data["Image Id"] + ".jpg"
        image_path = os.path.join(self.image_extract_path, "images", image_file)
        image = Image.open(image_path).convert("RGB")

        # Process prompt and response from conversation

        prompt = data["Prompt"]
        if self.add_image_prefix:
            prompt = f"<image> {prompt}"
        response = data["Rewritten Question"]
        if self.return_file_name:
            return image_file, prompt, response, image_file
        else:
            return image, prompt, response

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


class RevisionRewriteDatasetWithMetadata(Dataset):
    def __init__(
        self,
        dataset_name="anonymoususerrevision/multimodal_query_rewrites",
        split="train",
        filename_suffix="_with_metadata",
        use_auth_token=None,
        processor=None,
        add_image_prefix=False,
    ):
        self.processor = processor
        self.image_zip_path = hf_hub_download(
            repo_id=dataset_name,
            filename="images.zip",
            repo_type="dataset",
            use_auth_token=use_auth_token,
        )
        self.dataset_path = hf_hub_download(
            repo_id=dataset_name,
            filename=f"{split}{filename_suffix}.tsv",  # train or test
            repo_type="dataset",
            use_auth_token=use_auth_token,
        )

        with open(self.dataset_path) as f:
            self.dataset = pd.read_csv(self.dataset_path, sep="\t")
            print(f"length of data {len(self.dataset)}")

        self.image_extract_path = os.path.join(
            os.path.dirname(self.image_zip_path), "images"
        )

        if not os.path.exists(self.image_extract_path):
            print(f"Extracting image to {self.image_extract_path}")
            with zipfile.ZipFile(self.image_zip_path, "r") as zip_ref:
                zip_ref.extractall(self.image_extract_path)
        self.add_image_prefix = add_image_prefix

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the data entry
        data = self.dataset.iloc[idx]

        # Process image file path
        image_file = data["Image Id"] + ".jpg"

        image_path = os.path.join(self.image_extract_path, "images", image_file)
        image = Image.open(image_path).convert("RGB")

        # Process prompt and response from conversation
        prompt = data["Prompt"]
        response = data["Rewritten Question"]
        caption = data["Caption"]
        ocr_text = str(data["OCRText"])

        # Append Prompt with "<task>" tag
        if self.add_image_prefix:
            prompt = f"<image> {prompt}"
        prompt = "<task> " + prompt

        data_section = "<data> " + caption
        if len(ocr_text) > 0:
            data_section += " The text in the image is: " + ocr_text

        prompt_with_metadata = prompt + data_section

        return image, prompt_with_metadata, response

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


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        """
        Args:
            datasets (list of Dataset): A list of datasets to be combined.
        """
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = self._cumulative_sum(self.lengths)

    def _cumulative_sum(self, lengths):
        """Helper function to compute cumulative sums of lengths."""
        cum_sum = [0]
        for length in lengths:
            cum_sum.append(cum_sum[-1] + length)
        return cum_sum

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which dataset the idx falls into
        for i in range(len(self.datasets)):
            if idx < self.cumulative_lengths[i + 1]:
                return self.datasets[i][idx - self.cumulative_lengths[i]]
        raise IndexError("Index out of range.")

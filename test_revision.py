from model import ReVisionProcessor, ReVisionForConditionalGeneration
from datautils import RevisionRewriteDataset
from transformers import Trainer
from torch.utils.data import DataLoader
import pandas as pd
import torch
import argparse
import os
from tqdm import tqdm
from PIL import Image

# Model ID and auth token setup
MODEL_ID = "anonymoususerrevision/ReVision-250M-256-16-baseline"
use_auth_token = os.getenv("HF_TOKEN")

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def single_inference(processor, model, device, image_path, prompt):
    """Run single inference on an image and prompt."""
    image_raw = Image.open(image_path).convert("RGB")
    encoding = processor(images=image_raw, text=f"<input>{prompt}<rewrite>", return_tensors="pt")
    input_ids = encoding.pop("input_ids").to(device)
    image = encoding.pop("pixel_values").to(device)

    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, pixel_values=image, max_new_tokens=100)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    return generated_text

def collate_fn(self, examples, to_bf16=True):
    # Unpack the tuples into individual components
    images = [example[0].convert("RGB") for example in examples]
    texts = [example[1].replace("\n", "") for example in examples]  # Prompts
    labels = [example[2].replace("\n", "") for example in examples]  # Responses

    # Process with the processor
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the processor and model
    processor = ReVisionProcessor.from_pretrained(MODEL_ID, use_auth_token=use_auth_token, padding_side='left')
    model = ReVisionForConditionalGeneration.from_pretrained(MODEL_ID, use_auth_token=use_auth_token).to(device).eval()

    # Load the test dataset and DataLoader
    test_dataset = RevisionRewriteDataset(split="test", use_auth_token=use_auth_token, processor=processor)
    test_dataloader = DataLoader(
    	test_dataset,
    	batch_size=args.batch_size,
    	shuffle=False,
    	collate_fn=test_dataset.collate_fn  # Use the dataset's collate function
	)
	
	
    generated_texts = []

    print("Running inference on test set...")
    for batch_encoding in tqdm(test_dataloader):
        batch_encoding = {k: v.to(device) for k, v in batch_encoding.items()}
        
        with torch.inference_mode():
            output = model.generate(**batch_encoding, max_new_tokens=100)
        batch_generated_text = processor.batch_decode(output, skip_special_tokens=True)
        generated_texts.extend(batch_generated_text)

    # Load the ground truth data
    test_df = test_dataset.dataset.drop(columns = ["Unnamed: 0"])
    test_df["Predicted"] = pd.Series(generated_texts)

    # Save predictions to a new TSV file
    test_df.to_csv("test_predicted.tsv", sep='\t', index=False)

    print("Predictions saved to test_predicted.tsv")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned ReVision model on a test set.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for inference")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

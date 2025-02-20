import os
import glob
import easyocr
import pandas as pd
from tqdm import tqdm
import re
#! I truncated the OCRText text to 1024 characters post-script
THRESHOLD = 0.3

def main():
    images_dir = "images"
    output_dir = "easyocr_output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

    reader = easyocr.Reader(['en'])
    data_for_tsv = []
    for img_path in tqdm(image_files):        
        result = reader.readtext(img_path)

        image_text = ""
        for textbox, text, confidence in result:
            if confidence < THRESHOLD:
                continue

            image_text += text + " "
        
        image_text = re.sub(r'[^a-zA-Z0-9]', ' ', image_text)
        image_text = image_text.lower().strip()
        data_for_tsv.append([f"{img_path.split("/")[1]}", image_text])
    
    # Construct a DataFrame with columns as specified
    df = pd.DataFrame(data_for_tsv, columns=["image_name", "OCRText"])
    
    # Save as TSV in the root directory
    df.to_csv("easyocr_output.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()

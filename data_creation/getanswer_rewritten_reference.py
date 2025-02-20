from batchapi_runner import OpenAIBatchRunner
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import json
from getanswer_schema import SYSTEM_PROMPT, SCHEMA, BASE_TSV, validate_output, process_output_data_for_finetuned_models

def process_input_data(df: pd.DataFrame, processed_location: str) -> pd.DataFrame:
    if processed_location is None:
        print("Error: Input and Output processing needs `processed_location`")
        exit(1)

    new_df = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row["Image Id"]
        prompt = row["Rewritten Question"] # reference

        new_df.append({
            "id": f"{image_id}_{i}", # first part is the image name, second part is an index so that each question has a unique id
            "user_text_input": prompt
        })
    
    new_df = pd.DataFrame(new_df)

    if processed_location is not None:
        new_df.to_csv(processed_location, sep="\t", index=False)

    return new_df


if __name__ == "__main__":
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists("../logs/"):
        os.makedirs("../logs/")

    file_handler = logging.FileHandler("../logs/answers_reference.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    load_dotenv("./.env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    processed_file_location = "./TEMP_processed_test_reference.tsv"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, SYSTEM_PROMPT, json_schema=SCHEMA, input_file=BASE_TSV,
                               batch_folder="./batch_reference/")
    runner.create_jsonl_batches(process_input_data, processed_file_location, batch_size=250) # 250 batch size is ok without images
    runner.upload_batch_files()
    runner.submit_batch_jobs()
    runner.check_status_and_download()
    runner.fix_error_requests()
    runner.delete_data_files()

    output_data = runner.get_data()
    process_output_data_for_finetuned_models(output_data, processed_file_location, "qa_reference.tsv")

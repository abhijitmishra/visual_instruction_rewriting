from batchapi_runner import OpenAIBatchRunner
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import json
from getanswer_schema import SYSTEM_PROMPT, SCHEMA, BASE_TSV, validate_output

def process_input_data(df: pd.DataFrame, processed_location: str) -> pd.DataFrame:
    if processed_location is None:
        print("Error: Input and Output processing needs `processed_location`")
        exit(1)

    new_df = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row["Image Id"]
        prompt = row["Prompt"]

        # Remove the "Rewrite this..." prefix from the original prompt
        prompt = prompt.removeprefix("Rewrite this: ")
        prompt = prompt.removeprefix(
            "Rewrite this question based on image description: "
        )
        prompt = prompt.strip()

        new_df.append(
            {
                "id": f"{image_id}_{i}",  # first part is the image name, second part is an index so that each question has a unique id
                "user_text_input": prompt,
                "user_image_name": image_id,
            }
        )

    new_df = pd.DataFrame(new_df)

    if processed_location is not None:
        new_df.to_csv(processed_location, sep="\t", index=False)

    return new_df

def process_output_data(output_data: dict, processed_location: str, final_output_location: str):
    processed_df = pd.read_csv(processed_location, sep="\t")
    result = []
    for key, json_string in output_data.items():
        # connect key with prompt
        processed_row = processed_df[processed_df["id"] == key].squeeze()
        question_or_command = processed_row["user_text_input"]
        image_id = processed_row["user_image_name"]

        parts = key.split("_")
        image_id_from_key = "_".join(parts[:-1])
        row_index = int(parts[-1])

        # sanity check
        assert image_id == image_id_from_key

        try:
            response = json.loads(json_string)
            if validate_output(response):
                intent = response["intent"]
                arguments = json.dumps(response["arguments"])
            else:
                intent = "FIX"
                arguments = "{\"FIX\": true}"
        except Exception:
            print(f"Warning: Broken JSON for key={key}")
            print(json_string)
            intent = "FIX"
            arguments = "{\"FIX\": true}"

        result.append(
            {
                "Index": row_index,
                "Image Id": image_id_from_key,
                "Prompt": question_or_command,
                "Intent": intent,
                "Arguments": arguments
            }
        )


    # Create DataFrame, sort by the numeric index, and set the index column
    df = pd.DataFrame(result)
    df = df.sort_values("Index")
    df.set_index("Index", inplace=True)
    df.index.name = "Index"  # This will be used as the header for the index column in the TSV file

    df.to_csv(final_output_location, sep="\t", index=True)


if __name__ == "__main__":
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists("../logs/"):
        os.makedirs("../logs/")

    file_handler = logging.FileHandler("../logs/answers_original.log", mode="w")
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

    processed_file_location = "./TEMP_processed_test_original.tsv"
    runner = OpenAIBatchRunner(
        OPENAI_API_KEY,
        SYSTEM_PROMPT,
        json_schema=SCHEMA,
        input_file=BASE_TSV,
        batch_folder="./batch_original/",
    )
    runner.create_jsonl_batches(process_input_data, processed_file_location, batch_size=75)
    runner.upload_batch_files()
    runner.submit_batch_jobs()
    runner.check_status_and_download()
    runner.fix_error_requests()
    runner.delete_data_files()

    output_data = runner.get_data()
    process_output_data(output_data, processed_file_location, "qa_original.tsv")

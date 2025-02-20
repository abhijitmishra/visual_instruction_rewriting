from batchapi_runner import OpenAIBatchRunner
import logging
import os
from dotenv import load_dotenv
from getanswer_schema import SYSTEM_PROMPT, SCHEMA, process_input_data_for_finetuned_models, process_output_data_for_finetuned_models


if __name__ == "__main__":
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists("../logs/"):
        os.makedirs("../logs/")

    # file_handler = logging.FileHandler("../logs/answers_selfcaption_easyocr_8bit.log", mode="w")
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    load_dotenv("./.env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    input_file = "./results_selfcaption_easyocr_8bit.tsv"
    processed_file_location = "./TEMP_processed_test_selfcaption_easyocr_8bit.tsv"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, SYSTEM_PROMPT, json_schema=SCHEMA, input_file=input_file,
                               batch_folder="./batch_selfcaption_easyocr_8bit/")
    runner.create_jsonl_batches(process_input_data_for_finetuned_models, processed_file_location, batch_size=250) # 250 batch size is ok without images
    runner.upload_batch_files()
    runner.submit_batch_jobs()
    runner.check_status_and_download()
    runner.fix_error_requests()
    runner.delete_data_files()

    output_data = runner.get_data()
    process_output_data_for_finetuned_models(output_data, processed_file_location, "qa_selfcaption_easyocr_8bit.tsv")

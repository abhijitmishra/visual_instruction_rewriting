import os
from openai import OpenAI
import base64
import json
import logging
from copy import deepcopy
import time

class OpenAIBatchRunner:
    def __init__(
        self,
        openai_api_key: str,
        image_folder: str = "./images/",
        batch_input_folder: str = "./batch_input/",
        batch_output_folder: str = "./batch_output/",
        id_folder: str = "./ids/",
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.image_folder = image_folder
        self.image_file_names = self._get_image_files()

        # Directories for managing input/output and IDs
        self.batch_input_folder = batch_input_folder
        if not os.path.exists(batch_input_folder):
            os.makedirs(batch_input_folder)

        self.batch_output_folder = batch_output_folder
        if not os.path.exists(batch_output_folder):
            os.makedirs(batch_output_folder)

        self.id_folder = id_folder
        if not os.path.exists(id_folder):
            os.makedirs(id_folder)

        self.log = logging.getLogger(__name__)

        self.base_json = {
            "custom_id": None,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o-mini", "messages": None, "max_tokens": 1024},
        }

        self.system_prompt_caption = {
            "role": "system",
            "content": "You are an intelligent system generating descriptions for various images from the real world. "
            + "You will be provided with a single image, and your job will be to generate a descriptive caption.",
        }

        self.system_prompt_ocr = {
            "role": "system",
            "content": "You are a vision agent that can accurate read text from images. "
            + "You will be given a single image, and your job will be to report all legible text in a bulleted list, if any. "
            + "Do not report text that is not legible or too small to accurately read.",
        }

    def _get_image_files(self):
        """
        Retrieves the list of files from the image folder.
        Returns:
            List of file names
        """
        return [f for f in sorted(os.listdir(self.image_folder))]

    def encode_image(self, image_path):
        """
        Encodes the image into base64 format.
        Args:
            image_path (str): Path of the image to encode.
        Returns:
            str: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def create_jsonl_batches(self, batch_size: int = 50):
        """
        Create caption and OCR JSONL batch files. Suggested batch size is 50
        since each batch file is limited to 100 MB, and largest images in dataset
        are ~1MB. Image shrinking/cropping not necessary for costs since
        we are using low-resolution embedding.

        Args:
            batch_size (int): Number of responses in each JSONL file.
        """
        self.log.info("Creating JSONL batch files")
        num_batches = len(self.image_file_names) // batch_size + (
            1 if len(self.image_file_names) % batch_size != 0 else 0
        )

        # JSON reference
        caption_json = deepcopy(self.base_json)
        caption_json["body"]["messages"] = [self.system_prompt_caption, None]

        ocr_json = deepcopy(self.base_json)
        ocr_json["body"]["messages"] = [self.system_prompt_ocr, None]

        for batch_num in range(num_batches):
            caption_batch, ocr_batch = [], []
            start_idx, end_idx = batch_num * batch_size, (batch_num + 1) * batch_size

            image_name_batch = self.image_file_names[start_idx:end_idx]
            self.log.info(
                f"Creating batch {batch_num} from {start_idx} to {start_idx + len(image_name_batch) - 1}, inclusive"
            )

            for image_name in image_name_batch:
                image_path = os.path.join(self.image_folder, image_name)
                im_b64 = self.encode_image(image_path)

                caption_json["custom_id"] = f"caption_{image_name}"
                caption_json["body"]["messages"][1] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{im_b64}",
                                "detail": "low",
                            },
                        }
                    ],
                }
                caption_batch.append(deepcopy(caption_json))

                ocr_json["custom_id"] = f"ocr_{image_name}"
                ocr_json["body"]["messages"][1] = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{im_b64}",
                                "detail": "low",
                            },
                        }
                    ],
                }
                ocr_batch.append(deepcopy(ocr_json))

            self.log.info(
                f"    Created {len(caption_batch)} caption requests and {len(ocr_batch)} OCR requests"
            )
            self.log.info(f"    Writing batches to {self.batch_input_folder}")
            with open(
                f"{self.batch_input_folder}caption_{batch_num}.jsonl", mode="w"
            ) as caption_file:
                for request in caption_batch:
                    caption_file.write(json.dumps(request) + "\n")
            with open(
                f"{self.batch_input_folder}ocr_{batch_num}.jsonl", mode="w"
            ) as ocr_file:
                for request in ocr_batch:
                    ocr_file.write(json.dumps(request) + "\n")

        self.log.info(
            f"Finished writing all {num_batches} caption and {num_batches} OCR files"
        )

    def upload_batch_files(self):
        """
        Iterate through `self.batch_input_folder` and upload the files to OpenAI.
        Records OpenAI file IDs in `self.id_folder/fileids.txt`.
        """
        self.log.info("Uploading batch files...")
        openai_file_ids = []
        for batch_file_name in os.listdir(self.batch_input_folder):
            openai_file = self.client.files.create(
                file=open(f"{self.batch_input_folder}{batch_file_name}", mode="rb"),
                purpose="batch",
            )
            openai_file_ids.append((openai_file.id, openai_file.filename))
            self.log.info(f"Uploaded {openai_file.filename} with id: {openai_file.id}")
            time.sleep(1)

        self.log.info(f"Finished uploading {len(openai_file_ids)} files")
        with open(f"{self.id_folder}fileids.txt", mode="w") as f:
            for fileid, filename in openai_file_ids:
                f.write(f"{fileid}\t{filename}\n")

        self.log.info("Finished writing OpenAI file IDs locally")

    def submit_batch_jobs(self):
        """
        Submits batch jobs based on file IDs recorded in `self.id_folder`/fileids.txt.
        Records batch IDs locally in `self.id_folder/batchids.txt`
        """
        self.log.info("Submitting batch jobs...")
        file_ids = []
        with open(f"{self.id_folder}fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], file_entry[1].strip()))

        self.log.info(f"Retrieved {len(file_ids)} file ids from local file")
        batch_ids = []
        for file_id, file_name in file_ids:
            batch_job = self.client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_ids.append((batch_job.id, file_name))
            self.log.info(f"Submitted job for file {file_name} with ID: {batch_job.id}")
            time.sleep(1)

        self.log.info(f"Finished submitting {len(batch_ids)} jobs")

        with open(f"{self.id_folder}batchids.txt", mode="w") as f:
            for batch_id, file_name in batch_ids:
                f.write(f"{batch_id}\t{file_name}\n")
        self.log.info("Finished writing OpenAI file IDs locally")

    def check_status_and_download(self):
        """
        Read in the batch IDs from local file.
        Periodically checks for the batch job status.
        Downloads the resulting JSONL when complete.
        """
        self.log.info("Checking status for batch jobs...")
        batch_ids = []
        with open(f"{self.id_folder}batchids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                batch_ids.append((file_entry[0], file_entry[1].strip()))

        self.log.info(f"Retrieved {len(batch_ids)} batch IDs from local storage")

        # clear out the output_fileids.txt file
        with open(f"{self.id_folder}output_fileids.txt", mode="w") as f:
            f.write("")

        FAILED_STATUS = ["failed", "expired", "cancelled"]
        failed_batches = []
        while len(batch_ids) > 0:
            batch_indices_to_remove = []
            # go through the status of each batch job
            for i, (batch_id, file_name) in enumerate(batch_ids):
                job = self.client.batches.retrieve(batch_id)
                if job.status in FAILED_STATUS:
                    self.log.warning(
                        f'Batch with file {file_name} and ID "{batch_id}" has failed with status {job.status}'
                    )
                    failed_batches.append((batch_id, file_name))
                elif job.status == "completed":
                    self.log.info(
                        f"Batch for file {file_name} completed! Downloading data..."
                    )
                    result_file = self.client.files.content(job.output_file_id).content

                    with open(f"{self.id_folder}output_fileids.txt", mode="a") as f:
                        f.write(f"{job.output_file_id}\t{file_name}\n")

                    with open(
                        f"{self.batch_output_folder}output_{file_name}", mode="wb"
                    ) as f:
                        f.write(result_file)

                    batch_indices_to_remove.append(i)

                time.sleep(1)

            # remove batches from check list
            if len(batch_indices_to_remove) > 0:
                batch_ids = [
                    batch_id
                    for i, batch_id in enumerate(batch_ids)
                    if i not in batch_indices_to_remove
                ]

            if len(batch_ids) > 0:
                self.log.info("Sleeping for 5 minutes...")
                time.sleep(60.0 * 5.0)

        self.log.info("Finished retrieving data!")

    def delete_data_files(self):
        """
        Using the file_ids and batch_ids stored locally, delete them from OpenAI's file storage.
        """
        self.log.warning(
            "Starting deletion of input and output files stored in OpenAI's file storage..."
        )
        time.sleep(15)  # just in case you want to cancel

        file_ids = []
        with open(f"{self.id_folder}fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], file_entry[1].strip()))
        with open(f"{self.id_folder}output_fileids.txt", mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], "output_" + file_entry[1].strip()))

        self.log.info(f"Retrieved {len(file_ids)} file IDs")

        for file_id, name in file_ids:
            self.log.info(f"Deleting {name} with ID {file_id}")
            self.client.files.delete(file_id)
            time.sleep(2)

        self.log.info("Finished deleting files in OpenAI storage")


if __name__ == "__main__":
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")

    file_handler = logging.FileHandler("./logs/batchrunner.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    runner = OpenAIBatchRunner(OPENAI_API_KEY)
    runner.create_jsonl_batches()
    runner.upload_batch_files()
    runner.submit_batch_jobs()
    runner.check_status_and_download()
    runner.delete_data_files()

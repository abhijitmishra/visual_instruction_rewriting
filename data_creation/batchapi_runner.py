import os
from openai import OpenAI
import base64
import json
import logging
from copy import deepcopy
import time
import pandas as pd


class OpenAIBatchRunner:
    def __init__(
        self,
        openai_api_key: str,
        system_prompt: str,
        json_schema: dict | None = None,
        input_file: str = None,
        batch_folder: str = "../data/batch_temp/",
        image_folder: str = "./images/"
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.input_file = input_file
        self.image_folder = image_folder

        self.batch_folder = batch_folder

        # Directories for managing input/output and IDs
        self.batch_input_folder = os.path.join(batch_folder, "batch_input")
        if not os.path.exists(self.batch_input_folder):
            os.makedirs(self.batch_input_folder)

        self.batch_output_folder = os.path.join(batch_folder, "batch_output")
        if not os.path.exists(self.batch_output_folder):
            os.makedirs(self.batch_output_folder)

        self.batch_error_folder = os.path.join(batch_folder, "batch_error")
        if not os.path.exists(self.batch_error_folder):
            os.makedirs(self.batch_error_folder)

        self.id_folder = os.path.join(batch_folder, "ids")
        if not os.path.exists(self.id_folder):
            os.makedirs(self.id_folder)

        self.log = logging.getLogger(__name__)

        if json_schema:
            self.log.info("Utilizing structured output JSON Schema")
            self.base_json = {
                "custom_id": None,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": None,
                    "max_tokens": 1024,
                    "temperature": 0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": json_schema,
                    },
                },
            }
        else:
            self.base_json = {
                "custom_id": None,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o-mini", "messages": None, "temperature": 0, "max_tokens": 1024},
            }

        self.system_prompt = {
            "role": "system",
            "content": system_prompt,
        }

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

    def _validate_tsv(self, df: pd.DataFrame) -> bool:
        """
        Checks if the CSV file located at `file_path` includes the required columns
        ('id', 'user_text_input') and also checks for the presence of the optional
        column ('user_image_name').

        - If any required column is missing, prints an error message and exits.
        - Returns True if the optional column ('user_image_name') is present,
        otherwise returns False.
        """
        required_columns = ['id', 'user_text_input']
        optional_column = 'user_image_name'

        # Check for required columns
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                exit(1)

        # Return True if the optional column is found, otherwise False
        return optional_column in df.columns

    def create_jsonl_batches(
        self,
        processing_function = None,
        processed_file_location: str = None,
        batch_size: int = 50,
    ):
        """
        Create JSONL batch files. Suggested batch size is 50
        since each batch file is limited to 100 MB, and largest images in dataset
        are ~1MB. Image shrinking/cropping not necessary for costs since
        we are using low-resolution embedding.

        The input dataset must have 2 columns:
        - id
        - user_text_input

        with an optional 3rd column:
        - user_image_name

        If the `input_file` doesn't have these columns, provide a `processing_function` that converts the base dataframe object into this format.
        If the 3rd column is provided, the API will provide the base64 image to the OpenAI model.

        Args:
            batch_size (int): Number of responses in each JSONL file.
        """
        self.log.info("Creating JSONL batch files")
        df = pd.read_csv(self.input_file, delimiter="\t", index_col=False, keep_default_na=False)

        if processing_function:
            self.log.info("Processing input data using processing function...")
            df = processing_function(df, processed_file_location)

        # Ensure dataframe is expected
        use_image = self._validate_tsv(df)

        num_batches = df.shape[0] // batch_size + (
            1 if df.shape[0] % batch_size != 0 else 0
        )

        # JSON reference
        batch_json = deepcopy(self.base_json)
        batch_json["body"]["messages"] = [self.system_prompt, None]

        for batch_num in range(num_batches):
            batch_list = []
            start_idx, end_idx = batch_num * batch_size, (batch_num + 1) * batch_size

            df_batch = df[start_idx:end_idx]
            self.log.info(
                f"Creating batch {batch_num} from {start_idx} to {start_idx + df_batch.shape[0] - 1}, inclusive"
            )

            for _, item in df_batch.iterrows():
                batch_json["custom_id"] = item["id"]
                batch_json["body"]["messages"][1] = {
                    "role": "user",
                    "content": [{"type": "text", "text": str(item["user_text_input"])}],
                }

                # if the 3rd column is provided...
                if use_image:
                    image_name = item["user_image_name"]
                    image_path = os.path.join(self.image_folder, f"{image_name}.jpg")
                    im_b64 = self.encode_image(image_path)
                    batch_json["body"]["messages"][1]["content"].insert(0,
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{im_b64}",
                                "detail": "low",
                            },
                        }
                    )
                
                batch_list.append(deepcopy(batch_json))

            self.log.info(f"    Created {len(batch_list)} requests")
            self.log.info(f"    Writing batches to {self.batch_input_folder}")
            with open(
                os.path.join(self.batch_input_folder, f"batch_{batch_num}.jsonl"), mode="w"
            ) as caption_file:
                for request in batch_list:
                    caption_file.write(json.dumps(request) + "\n")

        self.log.info(f"Finished writing all {num_batches} files")

    def upload_batch_files(self):
        """
        Iterate through `self.batch_input_folder` and upload the files to OpenAI.
        Records OpenAI file IDs in `self.id_folder/fileids.txt`.
        """
        self.log.info("Uploading batch files...")
        openai_file_ids = []
        for batch_file_name in os.listdir(self.batch_input_folder):
            openai_file = self.client.files.create(
                file=open(os.path.join(self.batch_input_folder, batch_file_name), mode="rb"),
                purpose="batch",
            )
            openai_file_ids.append((openai_file.id, openai_file.filename))
            self.log.info(f"Uploaded {openai_file.filename} with id: {openai_file.id}")
            time.sleep(1)

        self.log.info(f"Finished uploading {len(openai_file_ids)} files")
        with open(os.path.join(self.id_folder, "fileids.txt"), mode="w") as f:
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
        with open(os.path.join(self.id_folder, "fileids.txt"), mode="r") as f:
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

        with open(os.path.join(self.id_folder, "batchids.txt"), mode="w") as f:
            for batch_id, file_name in batch_ids:
                f.write(f"{batch_id}\t{file_name}\n")
        self.log.info("Finished writing OpenAI file IDs locally")


    def cancel_all_batches(self, sleep_minutes: float = 3.0):
        """
        Cancels all batch submissions (can be used to make the code run the normal, non batch API in a hacky way)
        """
        self.log.info("Are you sure you want to cancel all batches?... Giving you 10 seconds.")
        time.sleep(10.0)

        batch_ids = []
        with open(os.path.join(self.id_folder, "batchids.txt"), mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                batch_ids.append((file_entry[0], file_entry[1].strip()))

        self.log.info("Checking if all batches are \"in_progress\"")
        not_in_progress_ids = deepcopy(batch_ids)
        while len(not_in_progress_ids) > 0:
            batch_indices_to_remove = []
            # go through the status of each batch job
            for i, (batch_id, file_name) in enumerate(batch_ids):
                job = self.client.batches.retrieve(batch_id)
                if job.status not in {"validating", "finalizing"}:
                    batch_indices_to_remove.append(i)

                time.sleep(0.5)
            # remove batches from check list
            if len(batch_indices_to_remove) > 0:
                not_in_progress_ids = [
                    batch_id
                    for i, batch_id in enumerate(not_in_progress_ids)
                    if i not in batch_indices_to_remove
                ]

            if len(not_in_progress_ids) > 0:
                self.log.info(f"Sleeping for {sleep_minutes} minutes...")
                time.sleep(60.0 * sleep_minutes)


        self.log.info(f"Cancelling all {len(batch_ids)} batches")
        for batch_id, file_name in batch_ids:
            self.log.info(f"Cancelling {file_name} batch")
            try:
                self.client.batches.cancel(batch_id)
            except Exception as e:
                self.log.exception(f"Skipping this batch: {e}")
            time.sleep(0.5)

        self.log.info("Now waiting 10 minutes to ensure all are `cancelled`...")
        time.sleep(60.0 * 10.0)
        self.log.info(f"Cancelled the batches. You must still download the batches that did finish to get all the results...")


    def check_status_and_download(self, sleep_minutes: float=5.0):
        """
        Read in the batch IDs from local file.
        Periodically checks for the batch job status.
        Downloads the resulting output/error JSONL files when complete.

        Note that if it passes 24 hours and some are still "in-progress", you should cancel those
        batches manually.

        Args:
            sleep_minutes (float): Number of minutes to sleep before requesting batch status
            resubmit_prompts (bool): (Only for text-only prompts) If true, requests data from OpenAI through standard API (not batch)
        """
        self.log.info("Checking status for batch jobs...")
        batch_ids = []
        with open(os.path.join(self.id_folder, "batchids.txt"), mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                batch_ids.append((file_entry[0], file_entry[1].strip()))

        self.log.info(f"Retrieved {len(batch_ids)} batch IDs from local storage")

        # clear out the output_fileids.txt file
        with open(os.path.join(self.id_folder, "output_fileids.txt"), mode="w") as f:
            f.write("")

        with open(os.path.join(self.id_folder, "error_fileids.txt"), mode="w") as f:
            f.write("")

        FAILED_STATUS = {"failed", "expired", "cancelled"}
        COMPLETED_STATUS = {"completed"}
        INPROGRESS_STATUS = {"validating", "in_progress", "finalizing", "cancelling"}

        while len(batch_ids) > 0:
            batch_indices_to_remove = []
            # go through the status of each batch job
            for i, (batch_id, file_name) in enumerate(batch_ids):
                job = self.client.batches.retrieve(batch_id)
                if job.status in INPROGRESS_STATUS:
                    continue
                elif job.status in FAILED_STATUS:
                    self.log.warning(
                        f'Batch with file {file_name} and ID "{batch_id}" has failed with status {job.status}'
                    )
                elif job.status in COMPLETED_STATUS:
                    self.log.info(
                        f"Batch for file {file_name} completed!"
                    )
                else:
                    self.log.error(f"Unknown status: {job.status}")
                    exit(1)

                batch_indices_to_remove.append(i)
                if job.output_file_id is not None:
                    self.log.info(f"Output file for {file_name} available! Downloading data...")
                    result_file = self.client.files.content(job.output_file_id).content
                    time.sleep(1)
                    
                    with open(
                        os.path.join(self.batch_output_folder, f"output_{file_name}"), mode="wb"
                    ) as f:
                        f.write(result_file)

                    with open(os.path.join(self.id_folder, "output_fileids.txt"), mode="a") as f:
                        f.write(f"{job.output_file_id}\t{file_name}\n")

                if job.error_file_id is not None:
                    self.log.info(
                            f"Error file for {file_name} available! Downloading data..."
                        )
                    result_file = self.client.files.content(job.error_file_id).content
                    time.sleep(1)
                    
                    with open(
                        os.path.join(self.batch_error_folder, f"error_{file_name}"), mode="wb"
                    ) as f:
                        f.write(result_file)

                    with open(os.path.join(self.id_folder, "error_fileids.txt"), mode="a") as f:
                        f.write(f"{job.error_file_id}\t{file_name}\n")

            # remove batches from check list
            if len(batch_indices_to_remove) > 0:
                batch_ids = [
                    batch_id
                    for i, batch_id in enumerate(batch_ids)
                    if i not in batch_indices_to_remove
                ]

            if len(batch_ids) > 0:
                self.log.info(f"Sleeping for {sleep_minutes} minutes...")
                time.sleep(60.0 * sleep_minutes)

        self.log.info("Finished retrieving data!")


    def fix_error_requests(self):
        """
        Iterates through requests in `self.batch_error_folder` through chat API (not batched)
        and appends the output folder with `error_fix_{batch name}.jsonl` files.
        """
        
        error_file_names = [
            f for f in sorted(os.listdir(self.batch_error_folder))
        ]

        self.log.info(f"Beginning resubmission of errored requests...")
        self.log.info(f"Found {len(error_file_names)} error batch files")

        for error_file_name in error_file_names:
            self.log.info(f"Processing {error_file_name}...")

            # get error custom_ids
            error_ids = []
            with open(os.path.join(self.batch_error_folder, error_file_name), mode="r") as f:
                for data_str in f:
                    data = json.loads(data_str)
                    error_ids.append(data["custom_id"])

            self.log.info(f"    Found {len(error_ids)} errored requests")

            # get input batch
            input_prompts = dict()
            input_file_name = error_file_name.removeprefix("error_")
            with open(os.path.join(self.batch_input_folder, input_file_name), mode="r") as f:
                for data_str in f:
                    data = json.loads(data_str)
                    
                    custom_id = data["custom_id"]
                    if custom_id not in error_ids:
                        continue

                    input_prompts[custom_id] = data["body"]
            
            # make sure all of the error ids is in input_prompts.keys()
            if len(input_prompts.keys()) != len(error_ids):
                self.log.warning(f"Mismatched keys and error ids for {error_file_name}: input has {len(input_prompts.keys())} and error has {len(error_ids)}")
                continue

            # make requests
            with open(os.path.join(self.batch_output_folder, f"error_fix_{input_file_name}"), mode="w") as f:
                for error_id in error_ids:
                    prompt_data = input_prompts[error_id]
                    
                    request = self.client.chat.completions.create(
                        model=prompt_data["model"],
                        messages=prompt_data["messages"],
                        max_tokens=prompt_data["max_tokens"],
                        temperature=prompt_data["temperature"],
                        response_format=prompt_data["response_format"]
                    )
                    new_json_entry = {
                        "id": "fix_error_request",
                        "custom_id": error_id,
                        "response": {
                            "status_code": 200, # I think the `chat.completions.create()` makes sure this is 200(?)
                            "request_id": "",   # purely to make `get_data()` work
                            "body": request.dict(),
                        },
                        "error": None
                    }

                    f.write(f"{json.dumps(new_json_entry)}\n")
    

    def run_batches_manual(self, input_batches: list):
        """
        Redos the batches and constructs `manual_batch_{batch_num}.jsonl` for those inputs
        """

        self.log.info(f"Running the following batches manually through non-batch API:\n{input_batches}")

        for input_batch in input_batches:
            # Retrieving input batch data
            input_prompts = dict()
            input_file_path = os.path.join(self.batch_input_folder, input_batch)
            with open(input_file_path, mode="r") as f:
                for data_str in f:
                    data = json.loads(data_str)
                    
                    custom_id = data["custom_id"]
                    input_prompts[custom_id] = data["body"]

            self.log.info(f"Retrieved {len(input_prompts)} from {input_file_path}")

            output_file_path = os.path.join(self.batch_output_folder, f"manual_{input_batch}")
            with open(output_file_path, mode="w") as f:
                for custom_id, prompt_data in input_prompts.items():
                    request = self.client.chat.completions.create(
                        model=prompt_data["model"],
                        messages=prompt_data["messages"],
                        max_tokens=prompt_data["max_tokens"],
                        temperature=prompt_data["temperature"],
                        response_format=prompt_data["response_format"]
                    )
                    new_json_entry = {
                        "id": "fix_error_request",
                        "custom_id": custom_id,
                        "response": {
                            "status_code": 200, # I think the `chat.completions.create()` makes sure this is 200(?)
                            "request_id": "",   # purely to make `get_data()` work
                            "body": request.dict(),
                        },
                        "error": None
                    }

                    f.write(f"{json.dumps(new_json_entry)}\n")


    def delete_data_files(self):
        """
        Using the file_ids and batch_ids stored locally, delete them from OpenAI's file storage.
        """
        self.log.warning(
            "Starting deletion of input and output files stored in OpenAI's file storage..."
        )
        time.sleep(15)  # just in case you want to cancel

        file_ids = []
        with open(os.path.join(self.id_folder, "fileids.txt"), mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], file_entry[1].strip()))
        with open(os.path.join(self.id_folder, "output_fileids.txt"), mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], "output_" + file_entry[1].strip()))
        with open(os.path.join(self.id_folder, "error_fileids.txt"), mode="r") as f:
            for data in f:
                file_entry = data.split("\t")
                file_ids.append((file_entry[0], "error_" + file_entry[1].strip()))

        self.log.info(f"Retrieved {len(file_ids)} file IDs")

        for file_id, name in file_ids:
            self.log.info(f"Deleting {name} with ID {file_id}")
            self.client.files.delete(file_id)
            time.sleep(1)

        self.log.info("Finished deleting files in OpenAI storage")

    def get_data(self, file_prefix: str = "batch") -> dict:
        """
        Retrieve JSONL data and returns dictionary of data in the form {custom_id: data}
        Args:
            batch_folder (str): Folder that stores output JSONL files
            file_prefix (str): String that is before each `_{batch_num}.jsonl` file
        """

        all_file_names = [
            f for f in sorted(os.listdir(self.batch_output_folder)) if file_prefix in f
        ]
        result_data = {}
        for file_name in all_file_names:
            file_path = os.path.join(self.batch_output_folder, file_name)
            with open(file_path, mode="r") as f:
                for line in f:
                    json_data = json.loads(line.rstrip())
                    obj_id = json_data["custom_id"]

                    if obj_id is None:
                        print(f"Obj Id is None for {file_path}")

                    response = json_data["response"]
                    if response["status_code"] != 200:
                        print(f"Warning! {obj_id} did not return response code 200")

                    choice = response["body"]["choices"][0]
                    output_content = choice["message"]["content"]
                    result_data[obj_id] = output_content

        return result_data

INTENT_LABELS = {
    "SetReminder", "SetAlarm", "CreateCalendarEvent", "SendMessage", "SendEmail",
    "MakeCall", "OpenApp", "SearchWeb", "SetTimer", "CheckWeather",
    "TurnOnDevice", "TurnOffDevice", "AdjustBrightness", "AdjustTemperature",
    "LockDoor", "UnlockDoor", "StartVacuum", "StopVacuum", "CheckSecurityCamera",
    "SetScene", "PlayMusic", "PauseMusic", "SkipTrack", "PlayPodcast", "PlayVideo",
    "AdjustVolume", "SetPlaybackSpeed", "SearchMovie", "ShowTVGuide",
    "GetDirections", "CheckTraffic", "FindNearbyPlace", "EstimateArrivalTime",
    "StartNavigation", "StopNavigation", "SendTextMessage", "MakePhoneCall",
    "StartVideoCall", "CheckVoicemail", "ReadMessage", "ReplyToMessage",
    "SendGroupMessage", "AnswerGeneralQuestion", "DefineWord", "ConvertUnits",
    "GetSportsScores", "CheckStockPrice", "GetFact", "TranslateText",
    "MathCalculation", "FindPersonInfo", "GetNewsUpdate"
}

ARGUMENT_LABELS = {
    "ReminderContent", "DateTime", "AlarmTime", "EventTitle", "EventLocation",
    "EventDateTime", "RecipientName", "MessageContent", "EmailSubject",
    "EmailBody", "AppName", "QueryText", "TimerDuration", "WeatherLocation",
    "WeatherDate", "DeviceName", "BrightnessLevel", "TemperatureValue",
    "SceneName", "LockState", "CameraLocation", "SongName", "ArtistName",
    "PodcastTitle", "EpisodeTitle", "VolumeLevel", "PlaybackSpeed", "MovieName",
    "TVChannel", "Destination", "CurrentLocation", "PlaceCategory", "ETA",
    "RouteType", "Recipient", "MessageBody", "ContactName", "VoicemailSender",
    "QuestionText", "WordToDefine", "UnitToConvert", "StockSymbol", "SportEvent",
    "PersonName", "LanguagePair", "MathExpression", "NewsTopic"
}

SCHEMA = {
  "name": "extracted_query",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "intent": {
        "type": "string",
        "enum": list(INTENT_LABELS)
      },
      "arguments": {
        "type": "object",
        "additionalProperties": False,
        "properties": { argument: {"type": ["string", "null"]} for argument in ARGUMENT_LABELS},
        "required": list(ARGUMENT_LABELS)
      }
    },
    "additionalProperties": False,
    "required": [
      "intent",
      "arguments"
    ]
  }
}

SYSTEM_PROMPT = """You are an advanced AI assistant that extracts structured information from natural language user queries. Given the following user query, determine the most appropriate intent from the predefined schema and extract relevant arguments with values.

Task:
  1. Identify the correct intent label from the list.
  2. Extract relevant arguments and their values from the query."""

# This is the test split of our dataset on huggingface
BASE_TSV = "./test.tsv"


import json
import pandas as pd
from tqdm import tqdm

def process_input_data_for_finetuned_models(df: pd.DataFrame, processed_location: str) -> pd.DataFrame:
    if processed_location is None:
        print("Error: Input and Output processing needs `processed_location`")
        exit(1)

    # Append "Image Id" and the rewrite prompt to the rewritten prompts
    test_df = pd.read_csv(BASE_TSV, sep="\t", keep_default_na=False)
    if len(test_df) != len(df):
        raise ValueError("Input TSV and test TSV must have the same number of rows")

    # This is kinda needed since we don't have a primary key on all the questions...
    if not test_df["Rewritten Question"].equals(df["Reference"]):
        raise ValueError("The \"Rewritten Question\" and \"Reference\" columns must match")

    df = pd.concat([
        test_df[["Image Id", "Prompt", "Rewritten Question"]],
        df[["Prediction"]]
    ], axis=1)

    df.rename(columns={
        "Prompt": "Initial Prompt",
        "Rewritten Question": "Rewritten Reference",
        "Prediction": "Rewritten Prediction"
    }, inplace=True)

    # Add Image Id column
    new_df = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row["Image Id"]
        prompt = row["Rewritten Prediction"]

        new_df.append({
            "id": f"{image_id}_{i}", # first part is the image name, second part is an index so that each question has a unique id
            "user_text_input": prompt,
            "Rewritten Reference": row["Rewritten Reference"], # Extra columns
            "Initial Prompt": row["Initial Prompt"] # Extra columns
        })
    
    new_df = pd.DataFrame(new_df)

    if processed_location is not None:
        new_df.to_csv(processed_location, sep="\t", index=False)

    return new_df


def validate_output(json_response: dict):
    intent_valid = "intent" in json_response and json_response["intent"] in INTENT_LABELS
    if not intent_valid:
        print(json.dumps(json_response))
        return False

    argument_valid = "arguments" in json_response and isinstance(json_response["arguments"], dict)
    if not argument_valid:
        print(json.dumps(json_response))
        return False
    
    for key in json_response["arguments"].keys():
        if key not in ARGUMENT_LABELS:
            print(json.dumps(json_response))
            return False
        
    return True

def process_output_data_for_finetuned_models(output_data: dict, processed_location: str, final_output_location: str):
    processed_df = pd.read_csv(processed_location, sep="\t")
    result = []
    
    for key, json_string in output_data.items():
        # Connect key with prompt
        processed_row = processed_df[processed_df["id"] == key].squeeze()
        question_or_command = processed_row["user_text_input"]

        # Extract image id and index
        parts = key.split("_")
        image_id_from_key = "_".join(parts[:-1])  # everything except the last element
        row_index = int(parts[-1])

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


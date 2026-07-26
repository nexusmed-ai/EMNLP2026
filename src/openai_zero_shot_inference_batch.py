"""Zero-shot binary life-threatening classification with Claude Sonnet via the
Anthropic Message Batches API.

Same batching approach as llm_soc_inference_batch.py: submits all requests
together via the async Batch API (POST /v1/messages/batches), which processes
up to 100k requests at 50% of standard per-token pricing instead of one
request at a time. Most batches finish within an hour; the hard cap is 24
hours.

Usage:
    export OPENAI_API_KEY=...
    python llm_severity_binary_inference_batch.py --output-file severity_sonnet_preds.jsonl
"""

import argparse
import json
import random
import re
import time
import pickle
import tempfile
import os

from openai import OpenAI
from dotenv import load_dotenv


# =========================
# Load OpenAI API Key
# =========================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)


# =========================
# Arguments
# =========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="gpt-5-mini"
)

parser.add_argument(
    "--oot-file",
    default="./adr_oot_new.pkl"
)

parser.add_argument(
    "--n-samples",
    type=int,
    default=1000
)

parser.add_argument(
    "--sample-seed",
    type=int,
    default=1234
)

parser.add_argument(
    "--output-file",
    default="severity_gpt5mini_preds.jsonl"
)

parser.add_argument(
    "--max-tokens",
    type=int,
    default=200
)

parser.add_argument(
    "--poll-seconds",
    type=int,
    default=30
)

args = parser.parse_args()


# =========================
# Load Dataset
# =========================

print("[1/4] Loading ADR cases")

adr_oot = pickle.load(
    open(args.oot_file, "rb")
)

import re


def normalize_adr_string(adr_string):
    """
    Convert malformed ADR pseudo-JSON string into valid JSON string.

    Example:
    {"patient":"{"age":"32"}","indi_pt":"depression"}

    becomes:

    {
      "patient": {"age":"32"},
      "indi_pt":"depression"
    }
    """

    if not isinstance(adr_string, str):
        return adr_string


    s = adr_string.strip()


    # Remove outer quotes if accidentally wrapped
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]


    # Fix escaped quotes if present
    s = s.replace('\\"', '"')


    # Detect quoted nested objects:
    # "patient":"{...}"
    pattern = r'"(\w+)":\s*"\{([^{}]*)\}"'


    def replace_nested(match):

        key = match.group(1)
        content = match.group(2)

        return f'"{key}":{{{content}}}'


    s = re.sub(
        pattern,
        replace_nested,
        s
    )


    # Try JSON parsing
    try:
        obj = json.loads(s)

        # Convert back to clean JSON
        return json.dumps(
            obj,
            ensure_ascii=False
        )

    except json.JSONDecodeError:

        # fallback: manually repair common issues

        s = re.sub(
            r'"\s*;',
            '"',
            s
        )

        return s

eval_idx = random.Random(
    args.sample_seed
).sample(
    range(len(adr_oot)),
    min(args.n_samples, len(adr_oot))
)

adr_tst = (
    adr_oot
    .iloc[eval_idx]
    .reset_index(drop=True)
)

adr_tst["inst"] = adr_tst["inst"].apply(normalize_adr_string)

print(f"    {len(adr_tst)} evaluation cases")


# =========================
# Prompt
# =========================

SYSTEM_PROMPT = """
You are a pharmacovigilance expert specializing in adverse drug reaction (ADR) risk assessment.

You will receive an ADR case report provided as a semi-structured string. The report does not contain the clinical outcome. 

First, parse the available clinical information, including:
- Patient demographics (e.g., age, sex, weight)
- Drug usage information (drug name(s), route(s) of administration, and dose(s), if available)
- Treatment indication(s)

Based solely on the information provided in the ADR case report, assess whether the case is likely to result in a life-threatening outcome or potentially fatal outcome.

Classification criteria:
- Return 1 if the case characteristics, drug exposure, or clinical information suggest a potential life-threatening outcome or risk of death.
- Return 0 if the available information does not indicate a potential life-threatening outcome.

If the information is insufficient to suggest a life-threatening risk, classify the case as 0.

Respond ONLY with valid JSON:

{"life_threatening": 1}

or

{"life_threatening": 0}

Do not provide explanations, reasoning, additional fields, or any text outside the JSON object.
"""


# =========================
# Create Batch File
# =========================

print(
    f"[2/4] Creating batch requests for {len(adr_tst)} cases"
)


with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False
) as fp:

    for i, inst_text in enumerate(adr_tst.inst):

        request = {

            "custom_id": str(i),

            "method": "POST",

            "url": "/v1/responses",

            "body": {

                "model": args.model,

                "input": [

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": inst_text
                    }

                ],

                # Important for GPT-5 models
                "reasoning": {
                    "effort": "minimal"
                },

                "max_output_tokens": args.max_tokens

            }
        }

        fp.write(
            json.dumps(request) + "\n"
        )


    batch_file = fp.name



# =========================
# Submit Batch
# =========================

uploaded = client.files.create(
    file=open(batch_file, "rb"),
    purpose="batch"
)


batch = client.batches.create(

    input_file_id=uploaded.id,

    endpoint="/v1/responses",

    completion_window="24h"

)


print(
    f"    Batch ID: {batch.id}"
)



# =========================
# Wait Completion
# =========================

print("[3/4] Waiting for batch completion")


while True:

    batch = client.batches.retrieve(
        batch.id
    )

    print(
        "    status:",
        batch.status
    )


    if batch.status == "completed":
        break


    if batch.status in (
        "failed",
        "expired",
        "cancelled"
    ):
        raise RuntimeError(
            batch.status
        )


    time.sleep(
        args.poll_seconds
    )



# =========================
# Parse Responses
# =========================


def extract_response_text(obj):

    body = obj["response"]["body"]

    if body.get("status") != "completed":
        print(
            "FAILED:",
            body.get("incomplete_details")
        )
        return ""

    # New Responses API format
    if "output_text" in body:
        return body["output_text"]


    for item in body.get("output", []):

        if item.get("type") == "message":

            for c in item.get("content", []):

                if "text" in c:
                    return c["text"]


    return ""



def parse_prediction(text):

    if not text:
        return None


    try:

        obj = json.loads(
            text.strip()
        )

        return int(
            obj["life_threatening"]
        )


    except Exception:


        match = re.search(
            r'"life_threatening"\s*:\s*(0|1)',
            text
        )


        if match:

            return int(
                match.group(1)
            )


    return None



print(
    "[4/4] Collecting results"
)


result = client.files.content(
    batch.output_file_id
)


generated_by_id = {}


for line in result.text.strip().splitlines():

    obj = json.loads(line)

    cid = obj["custom_id"]

    text = extract_response_text(
        obj
    )

    generated_by_id[cid] = text



# =========================
# Save Predictions
# =========================


with open(
    args.output_file,
    "w"
) as f:


    none_count = 0


    for i, inst_text in enumerate(adr_tst.inst):

        raw = generated_by_id.get(
            str(i),
            ""
        )


        prediction = parse_prediction(
            raw
        )


        if prediction is None:
            none_count += 1


        f.write(
            json.dumps(
                {
                    "inst": inst_text,
                    "prediction": prediction,
                    "raw_response": raw
                }
            )
            + "\n"
        )



print(
    f"Completed. Saved {len(adr_tst)} records"
)

print(
    f"Missing predictions: {none_count}"
)


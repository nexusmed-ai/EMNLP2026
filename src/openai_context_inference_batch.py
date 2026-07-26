"""RAG-context binary life-threatening classification with an OpenAI model via
the Batch API.

Loads llm_context_prompt.pkl (query case + top5 BM25 + top5 vector retrievals,
each retrieval paired with its known 0/1 severity label) and builds a
few-shot prompt per case: the top10 retrieved cases are shown as labeled
user/assistant demonstration turns, followed by the query case to classify.

Usage:
    export OPENAI_API_KEY=...
    python openai_severity_binary_inference_batch.py --output-file severity_gpt5mini_rag_preds.jsonl
"""

import argparse
import json
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
    "--context-file",
    default="./llm_context_prompt.pkl",
    help="pkl with 5 parts: query inst, top5 BM25 inst, top5 BM25 labels, top5 vector inst, top5 vector labels"
)

parser.add_argument(
    "--n-samples",
    type=int,
    default=None,
    help="number of cases to run, taken from the start of context-file (default: all)"
)

parser.add_argument(
    "--output-file",
    default="severity_gpt5mini_rag_preds.jsonl"
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

print("[1/4] Loading context prompts")

query_inst, bm25_inst, bm25_label, vect_inst, vect_label = pickle.load(
    open(args.context_file, "rb")
)

if args.n_samples is not None:
    query_inst = query_inst[:args.n_samples]
    bm25_inst = bm25_inst[:args.n_samples]
    bm25_label = bm25_label[:args.n_samples]
    vect_inst = vect_inst[:args.n_samples]
    vect_label = vect_label[:args.n_samples]

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

query_inst = [normalize_adr_string(t) for t in query_inst]
bm25_inst = [[normalize_adr_string(t) for t in texts] for texts in bm25_inst]
vect_inst = [[normalize_adr_string(t) for t in texts] for texts in vect_inst]

print(f"    {len(query_inst)} evaluation cases, "
      f"{len(bm25_inst[0])} BM25 + {len(vect_inst[0])} vector retrievals each")


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

After the case to classify, you will be shown up to 10 reference ADR case reports retrieved for their clinical similarity to it, each labeled with its known life-threatening outcome (0 or 1). Treat these only as supporting evidence. Base your prediction solely on the case to classify and this evidence -- do not use any other outside knowledge.

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
# Build per-case prompt: query inst + top10 evidence pairs
# =========================

def build_input_messages(i):

    retrieved_inst = bm25_inst[i] + vect_inst[i]
    retrieved_label = bm25_label[i] + vect_label[i]

    evidence_block = "\n\n".join(
        f"Evidence {j} (life_threatening={ref_label}):\n{ref_text}"
        for j, (ref_text, ref_label) in enumerate(zip(retrieved_inst, retrieved_label), start=1)
    )

    user_content = (
        f"Case to classify:\n{query_inst[i]}\n\n"
        f"Ten relevant evidence cases with known outcomes:\n{evidence_block}"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    return messages


# =========================
# Create Batch File
# =========================

print(
    f"[2/4] Creating batch requests for {len(query_inst)} cases"
)


with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False
) as fp:

    for i in range(len(query_inst)):

        request = {

            "custom_id": str(i),

            "method": "POST",

            "url": "/v1/responses",

            "body": {

                "model": args.model,

                "input": build_input_messages(i),

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


    for i, inst_text in enumerate(query_inst):

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
    f"Completed. Saved {len(query_inst)} records"
)

print(
    f"Missing predictions: {none_count}"
)


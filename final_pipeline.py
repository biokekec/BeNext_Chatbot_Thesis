# Closed-book QA pipeline only
# Select one model: "qwen" or "mistral"

%pip install -q -U "transformers==4.45.0" "accelerate" "bitsandbytes>=0.46.1" "sentencepiece" "lingua-language-detector"

from pathlib import Path
import json
import time
import gc
import pandas as pd
import torch
import transformers
import accelerate
import bitsandbytes as bnb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

from lingua import Language, LanguageDetectorBuilder


# ============================================================
# 1. Model selection
# ============================================================

MODEL_OPTIONS = {
    "qwen_small": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_big": "Qwen/Qwen2.5-14B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

MODEL_CHOICE = "qwen_small"  
CURRENT_MODEL_ID = MODEL_OPTIONS[MODEL_CHOICE]


# ============================================================
# 2. Load guidelines
# ============================================================

DATA_DIR = Path("/content")
GUIDELINES_FILE = DATA_DIR / "QA_guidelines.csv"

guidelines_df = pd.read_csv(GUIDELINES_FILE)

required_guideline_columns = [
    "guideline_type",
    "guideline_text",
]

missing_columns = [
    col for col in required_guideline_columns
    if col not in guidelines_df.columns
]

if missing_columns:
    raise ValueError(f"Guidelines dataframe is missing required columns: {missing_columns}")

guidelines_df["guideline_type"] = (
    guidelines_df["guideline_type"]
    .fillna("")
    .astype(str)
    .str.strip()
)

guidelines_df["guideline_text"] = (
    guidelines_df["guideline_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

guideline_dict = (
    guidelines_df
    .groupby("guideline_type")["guideline_text"]
    .apply(list)
    .to_dict()
)


def format_guidelines(guideline_dict: dict) -> str:
    guideline_sections = []

    for guideline_type, guideline_texts in guideline_dict.items():
        guideline_sections.append(str(guideline_type))

        for guideline_text in guideline_texts:
            guideline_sections.append(f"- {guideline_text}")

        guideline_sections.append("")

    return "\n".join(guideline_sections).strip()


# ============================================================
# 3. Language detection
# ============================================================

lingua_detector = (
    LanguageDetectorBuilder
    .from_languages(Language.ENGLISH, Language.DUTCH)
    .build()
)


def detect_language(user_question: str) -> str:
    detected_language = lingua_detector.detect_language_of(user_question)

    if detected_language == Language.DUTCH:
        return "nl"

    if detected_language == Language.ENGLISH:
        return "en"

    return "en"


# ============================================================
# 4. Model loading
# ============================================================

def load_generation_model(model_id: str):
    print("=" * 80)
    print(f"Loading model: {model_id}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)

    model.eval()

    print("Model loaded:", model_id)

    return tokenizer, model, device


def unload_generation_model():
    global model
    global tokenizer

    try:
        del model
    except NameError:
        pass

    try:
        del tokenizer
    except NameError:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Model unloaded and GPU cache cleared.")


tokenizer, model, device = load_generation_model(CURRENT_MODEL_ID)


# ============================================================
# 5. Generation helpers
# ============================================================

def generate_text_with_model(
    prompt: str,
    max_new_tokens: int = 120,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted_prompt = prompt

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    input_length = inputs["input_ids"].shape[-1]

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.do_sample = False
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )

    generated_ids = output_ids[0][input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return answer


def parse_json_from_text(raw_output: str) -> dict:
    try:
        return json.loads(raw_output)
    except Exception:
        pass

    try:
        start_index = raw_output.find("{")
        end_index = raw_output.rfind("}") + 1

        if start_index != -1 and end_index != -1:
            json_text = raw_output[start_index:end_index]
            return json.loads(json_text)
    except Exception:
        pass

    return {}


def clean_generated_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return ""

    cleaned_answer = answer.strip()

    unwanted_prefixes = [
        "Sure!",
        "Sure,",
        "Of course!",
        "Of course,",
        "Let's break down",
        "Let’s break down",
        "Here is the answer:",
        "Here is the response:",
        "Final answer:",
        "Final chatbot answer:",
        "Answer:",
        "Hier is de juiste antwoord:",
        "Hier is het juiste antwoord:",
        "Hier is het antwoord:",
        "Natuurlijk!",
        "Natuurlijk,",
        "Zeker!",
        "Zeker,",
    ]

    for prefix in unwanted_prefixes:
        if cleaned_answer.lower().startswith(prefix.lower()):
            cleaned_answer = cleaned_answer[len(prefix):].strip()

    unwanted_headings = [
        "### System Context:",
        "### Reasoning:",
        "### Explanation:",
        "### Answer:",
        "## System Context:",
        "## Reasoning:",
        "## Explanation:",
        "## Answer:",
        "System Context:",
        "Reasoning:",
        "Explanation:",
    ]

    for heading in unwanted_headings:
        cleaned_answer = cleaned_answer.replace(heading, "").strip()

    forbidden_identity_terms = [
        "Qwen",
        "Hugging Face",
        "OpenAI",
        "LLM",
        "large language model",
        "AI model",
        "language model",
        "prompt",
        "retrieved examples",
        "benchmark",
        "internal pipeline",
    ]

    for term in forbidden_identity_terms:
        cleaned_answer = cleaned_answer.replace(term, "BeNext energy support assistant")

    cleaned_answer = cleaned_answer.replace("```", "").strip()
    cleaned_answer = " ".join(cleaned_answer.split())

    return cleaned_answer


# ============================================================
# 6. Scope detection
# ============================================================

def detect_scope(user_question: str) -> str:
    scope_prompt = f"""
You are a routing classifier for a BeNext household energy monitoring chatbot.

Classify the user question as exactly one of these labels:

1. in_scope
The question is about household energy monitoring, electricity use, energy consumption, solar production, grid export, heat pump behaviour, ventilation, water use, device-related energy behaviour, energy graphs, peaks, anomalies, or follow-up recommendations based on system readings.

2. out_of_scope
The question is about anything else. This includes recipes, stocks, medical advice, politics, travel, coding, movies, translation, jokes, Wi-Fi/router support, account/login support, legal advice, definitive billing advice, tariff advice, compensation advice, device repair instructions, installer-level troubleshooting, or housing-provider disputes.

User question:
{user_question}

Return only valid JSON with this exact structure:
{{
  "scope": "in_scope",
  "reason": "Brief reason."
}}
""".strip()

    raw_output = generate_text_with_model(
        prompt=scope_prompt,
        max_new_tokens=80,
    )

    parsed_output = parse_json_from_text(raw_output)

    scope = parsed_output.get("scope", "").strip().lower()

    if scope in ["in_scope", "out_of_scope"]:
        return scope

    return "out_of_scope"


def get_out_of_scope_response(language: str) -> str:
    if language == "nl":
        return (
            "Ik kan alleen helpen met vragen over energiemonitoring in huis, "
            "energieverbruik, zonne-energie, warmtepompen, ventilatie en afwijkende energiepatronen. "
            "Stel gerust een vraag over uw energiegegevens of systeemmetingen."
        )

    return (
        "I can only help with questions about household energy monitoring, "
        "energy use, solar production, heat pumps, ventilation, and related anomalies. "
        "Please ask a question about your energy data or system readings."
    )


# ============================================================
# 7. Closed-book prompt
# ============================================================

def build_closed_book_prompt(
    user_question: str,
    language: str,
    system_context: str = "",
) -> str:
    if language not in ["en", "nl"]:
        raise ValueError("Supported languages are only 'en' and 'nl'.")

    language_name = {
        "en": "English",
        "nl": "Dutch",
    }[language]

    guidelines_text = format_guidelines(guideline_dict)

    if not system_context:
        system_context = "No additional system-side context is available for this question."

    prompt = f"""
You are a BeNext energy support assistant for household energy monitoring.

This is a closed-book QA setting. Do not use retrieved examples or external documents.
Answer only the current user question using the current system-side context and the answer guidelines.

Answer language:
{language_name}

General answer guidelines:
{guidelines_text}

Current system-side context:
{system_context}

Current user question:
{user_question}

Answer requirements:
- Output only the final user-facing chatbot answer.
- Do not include headings, bullet points, markdown, labels, or step-by-step reasoning.
- Do not start with phrases such as "Sure", "Let's break down", "Here is the answer", or similar.
- Answer in {language_name}.
- Write 2 to 4 short sentences.
- Use simple, non-technical language.
- Use only the current system-side context as evidence.
- Do not invent causes.
- If the exact cause is not confirmed, say that clearly.
- Give at most one practical next step.
- Avoid alarmist wording.
- Never mention Qwen, Hugging Face, OpenAI, LLM, AI model, prompt, benchmark data, or internal pipeline.
- If asked what you are, refer to yourself only as a BeNext energy support assistant.

Final chatbot answer only:
""".strip()

    return prompt


# ============================================================
# 8. Closed-book QA
# ============================================================

def generate_closed_book_answer(
    user_question: str,
    language: str = "",
    system_context: str = "",
    max_new_tokens: int = 90,
    forced_scope: str = "",
    store_prompt: bool = False,
) -> dict:
    start_time = time.perf_counter()

    if not language:
        language = detect_language(user_question)

    detected_language = language

    if forced_scope:
        detected_scope = forced_scope
        scope_detection_method = "cached_scope"
    else:
        detected_scope = detect_scope(user_question)
        scope_detection_method = "prompt_based"

    if detected_scope == "out_of_scope":
        answer = get_out_of_scope_response(language)
        end_time = time.perf_counter()

        return {
            "approach": "closed_book",
            "pipeline_model_key": MODEL_CHOICE,
            "pipeline_model_id": CURRENT_MODEL_ID,
            "user_question": user_question,
            "language": language,
            "system_context": system_context,
            "prompt": "",
            "generated_answer": answer,
            "latency_seconds": end_time - start_time,
            "detected_scope": detected_scope,
            "scope_detection_method": scope_detection_method,
            "detected_language": detected_language,
            "language_detection_method": "lingua",
        }

    prompt = build_closed_book_prompt(
        user_question=user_question,
        language=language,
        system_context=system_context,
    )

    answer = generate_text_with_model(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    answer = clean_generated_answer(answer)

    end_time = time.perf_counter()

    return {
        "approach": "closed_book",
        "pipeline_model_key": MODEL_CHOICE,
        "pipeline_model_id": CURRENT_MODEL_ID,
        "user_question": user_question,
        "language": language,
        "system_context": system_context,
        "prompt": prompt if store_prompt else "",
        "generated_answer": answer,
        "latency_seconds": end_time - start_time,
        "detected_scope": detected_scope,
        "scope_detection_method": scope_detection_method,
        "detected_language": detected_language,
        "language_detection_method": "lingua",
    }


# ============================================================
# 9. Batch generation
# ============================================================

def generate_closed_book_answers_for_dataframe(
    input_df: pd.DataFrame,
    max_new_tokens: int = 90,
    store_prompt: bool = False,
) -> pd.DataFrame:
    generated_rows = []

    total_items = len(input_df)

    print(
        f"Starting closed-book QA generation for {total_items} items | "
        f"model={MODEL_CHOICE} | "
        f"model_id={CURRENT_MODEL_ID}"
    )

    for row_number, (_, row) in enumerate(input_df.iterrows(), start=1):
        item_id = row.get("item_id", f"row_{row_number}")
        dataset = row.get("dataset", "")
        true_language = row.get("language", "")
        question = row.get("question", "")
        context = row.get("context", "")
        forced_scope = row.get("cached_detected_scope", "")

        print(f"Generating {row_number}/{total_items}: {item_id}")

        try:
            result = generate_closed_book_answer(
                user_question=question,
                language="",
                system_context=context,
                max_new_tokens=max_new_tokens,
                forced_scope=forced_scope,
                store_prompt=store_prompt,
            )

            error_message = ""

        except Exception as error:
            result = {
                "approach": "closed_book",
                "pipeline_model_key": MODEL_CHOICE,
                "pipeline_model_id": CURRENT_MODEL_ID,
                "user_question": question,
                "language": "",
                "system_context": context,
                "prompt": "",
                "generated_answer": "",
                "latency_seconds": None,
                "detected_scope": "",
                "scope_detection_method": "",
                "detected_language": "",
                "language_detection_method": "lingua",
            }

            error_message = str(error)

        output_row = {
            "item_id": item_id,
            "dataset": dataset,
            "true_language": true_language,
            "question": question,
            "context": context,
            "reference_answer": row.get("reference_answer", ""),
            "expected_answer_points": row.get("expected_answer_points", ""),
            "should_not_include": row.get("should_not_include", ""),
            "source_basis": row.get("source_basis", ""),
            **result,
            "language_detection_correct": (
                result.get("detected_language", "") == true_language
                if true_language
                else None
            ),
            "error_message": error_message,
        }

        generated_rows.append(output_row)

    results_df = pd.DataFrame(generated_rows)

    print("Closed-book QA generation finished.")
    print("Results shape:", results_df.shape)

    return results_df


# ============================================================
# 10. Run final closed-book pipeline
# ============================================================

closed_book_results = generate_closed_book_answers_for_dataframe(
    input_df=test_df,
    max_new_tokens=90,
    store_prompt=False,
)

closed_book_output_path = DATA_DIR / f"closed_book_QA_results_{MODEL_CHOICE}.csv"
closed_book_results.to_csv(closed_book_output_path, index=False)

print("Closed-book QA results saved to:")
print(closed_book_output_path)

display(closed_book_results.head())

# Multilingual QA for Energy Support

This repository contains the code, constructed benchmark datasets, model outputs, and evaluation files used for a thesis project on multilingual question answering for residential energy support. The project compares different chatbot approaches for answering user questions about household energy usage, energy monitoring, solar production, heat pumps, ventilation, and unusual energy patterns.

The work was developed in the context of a BeNext chatbot prototype and evaluates three main QA approaches:

1. **Closed-book QA**
   The model answers using only the user question, available system-side context, and answer guidelines.

2. **Retrieval-based QA**
   The system retrieves the most similar existing QA case and returns the corresponding reference answer as a nearest-neighbour baseline.

3. **Retrieval-augmented generation (RAG)**
   The system retrieves similar QA cases and provides them as examples in the prompt before generating a new answer.

## Repository contents

```text
.
├── language_scope_detection_test.ipynb
├── pipeline_test.ipynb
├── final_pipeline.py
├── data/
│   ├── in_scope_QA.csv
│   ├── out_of_scope_QA.csv
│   └── QA_guidelines.csv
├── results/
│   ├── qa_results_Qwen_Qwen2.5-7B-Instruct.csv
│   ├── qa_results_Qwen_Qwen2.5-14B-Instruct.csv
│   ├── qa_results_mistralai_Mistral-7B-Instruct-v0.3.csv
│   ├── qa_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B.csv
│   ├── qa_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-14B.csv
│   └── evaluation/
│       ├── benext_llm_judge_evaluated_answers.csv
│       ├── benext_llm_judge_summary_by_approach_dataset.csv
│       ├── benext_llm_judge_summary_by_group.csv
│       ├── annotation_1.xlsx
│       └── annotation_2.xlsx
├── .gitignore
└── README.md
```

## Files

### `language_scope_detection_test.ipynb`

This notebook tests the routing layer of the chatbot pipeline.

It evaluates:

* language detection for English and Dutch questions;
* scope detection for identifying whether a question belongs to the supported energy-monitoring domain;
* rule-based versus prompt-based scope detection;
* Lingua-based versus prompt-based language detection.

The purpose of this notebook is to validate the routing choices before running full answer-generation experiments.

### `pipeline_test.ipynb`

This notebook contains the experimental pipeline used to test and compare the QA approaches.

It includes:

* loading and validating the constructed datasets;
* splitting in-scope cases into a retrieval library and held-out test set;
* constructing bilingual benchmark items;
* preparing a TF-IDF retrieval baseline;
* building prompts for closed-book QA and RAG;
* applying language and scope routing;
* generating answers with selected open-source language models;
* saving model outputs and metadata for later evaluation.

This file should be treated as the main experimental notebook.

### `final_pipeline.py`

This file contains the cleaned final version of the pipeline with the final model and approach choices.

It is intended to provide a more reproducible version of the notebook workflow for further use, without unnecessary notebook outputs, manual display calls, or Colab-specific experimentation.

## Data

The `data/` folder contains the constructed benchmark datasets.

### `data/in_scope_QA.csv`

Contains constructed in-scope QA pairs related to residential energy support. Each row includes an English question, Dutch question, system-side context, reference answers, expected answer points, and content that should not be included in the chatbot answer.

The in-scope examples cover topics such as household electricity use, solar production, heat pump behaviour, ventilation, energy monitoring, unusual energy patterns, and practical interpretation of system readings.

### `data/out_of_scope_QA.csv`

Contains constructed out-of-scope examples. These are used to test whether the chatbot correctly refuses questions outside the supported domain, such as unrelated general questions, legal advice, device repair instructions, login support, billing advice, tariff advice, compensation questions, or other unsupported requests.

### `data/QA_guidelines.csv`

Contains answer guidelines used in the chatbot prompts. These guidelines define the desired answer style, scope boundaries, and safety constraints.

The guidelines instruct the chatbot to use simple language, avoid unsupported claims, avoid technical or alarmist wording, stay within the BeNext energy-monitoring domain, and give at most one practical next step when appropriate.

## Results

The `results/` folder contains the generated chatbot answers from the final model runs, together with the evaluation files used to analyse answer quality.

Each model result file contains the outputs for one model across the three QA approaches:

* closed-book QA;
* retrieval-based QA;
* retrieval-augmented generation.

Each result file contains generated answers and metadata such as item ID, dataset type, language, detected language, detected scope, generated answer, retrieved case IDs, similarity scores, latency, and model ID.

The following model result files are included:

### `results/qa_results_Qwen_Qwen2.5-7B-Instruct.csv`

Contains the generated outputs for Qwen2.5-7B-Instruct.

### `results/qa_results_Qwen_Qwen2.5-14B-Instruct.csv`

Contains the generated outputs for Qwen2.5-14B-Instruct.

### `results/qa_results_mistralai_Mistral-7B-Instruct-v0.3.csv`

Contains the generated outputs for Mistral-7B-Instruct-v0.3.

### `results/qa_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B.csv`

Contains the generated outputs for DeepSeek-R1-Distill-Qwen-7B.

### `results/qa_results_deepseek-ai_DeepSeek-R1-Distill-Qwen-14B.csv`

Contains the generated outputs for DeepSeek-R1-Distill-Qwen-14B.

## Evaluation files

The `results/evaluation/` folder contains the LLM-as-a-judge evaluation outputs and manual annotation files.

### `results/evaluation/benext_llm_judge_evaluated_answers.csv`

Contains the LLM-as-a-judge scores for the generated chatbot answers. The file includes the original answer metadata together with evaluation scores for:

* factual accuracy;
* grounding;
* clarity;
* usefulness;
* language correctness;
* scope handling;
* overall quality;
* judge notes.

This is the main evaluation file used for analysing QA approach performance, model performance, language differences, and metric-level differences.

### `results/evaluation/benext_llm_judge_summary_by_approach_dataset.csv`

Contains aggregated LLM-judge results grouped by QA approach and dataset type. This file is useful for comparing performance on in-scope versus out-of-scope questions.

### `results/evaluation/benext_llm_judge_summary_by_group.csv`

Contains grouped summary statistics from the LLM-judge evaluation. This file is used for higher-level comparison of approaches, models, languages, and evaluation metrics.

### `results/evaluation/annotation_1.xlsx`

Contains manual annotation data for a sample of generated answers. This file is used to support manual checking of answer quality and to compare human judgement with automated LLM-as-a-judge scores.

### `results/evaluation/annotation_2.xlsx`

Contains an additional manual annotation file. This file supports validation of the automated evaluation process and comparison between annotation rounds or annotators.

## Experimental setup

The benchmark consists of 50 constructed in-scope QA pairs and 30 constructed out-of-scope QA pairs. Each pair is available in English and Dutch.

To avoid retrieval leakage, the 50 in-scope QA pairs were split before bilingual expansion. A 70/30 split was used:

* 35 in-scope QA pairs were used for the retrieval library;
* 15 in-scope QA pairs were held out for testing.

After bilingual expansion, this produced:

* 70 retrieval items;
* 30 held-out in-scope test items;
* 60 out-of-scope test items.

The final test set therefore contains 90 bilingual test items.

The final model comparison includes five open instruction-tuned models:

* Qwen2.5-7B-Instruct;
* Qwen2.5-14B-Instruct;
* Mistral-7B-Instruct-v0.3;
* DeepSeek-R1-Distill-Qwen-7B;
* DeepSeek-R1-Distill-Qwen-14B.

The final pipeline uses Lingua for language detection and prompt-based scope detection for deciding whether a question is inside or outside the supported energy-monitoring domain. Scope routing was fixed across model runs so that answer-generation models could be compared under the same routing conditions.

## QA approaches

### Closed-book QA

Closed-book QA generates an answer using only the current user question, the available system-side context, and the general answer guidelines. It does not use retrieved examples.

This approach is included as the simplest generative baseline.

### Retrieval-based QA

Retrieval-based QA uses TF-IDF similarity to retrieve the most similar question from the retrieval library. It then returns the corresponding reference answer or a fallback response.

This approach does not generate a new answer with the answer-generation model. It is included as a nearest-neighbour baseline.

### Retrieval-augmented generation

RAG first retrieves similar cases from the retrieval library and then provides them to the model as examples in the prompt. The model then generates a new answer for the current question.

In the RAG setup, retrieval is based on question-level TF-IDF similarity. Retrieved reference answers are only included when the similarity score is high enough. This was done to reduce the risk of copying answers from weakly related cases.

## Notes on retrieval-based QA

Retrieval-based QA is included as a fixed nearest-neighbour baseline. It does not use the answer-generation model to create a new response.

Because retrieval-based QA does not depend on the answer-generation model, it should not be interpreted as a model-specific generative result. It is included to compare retrieval-only behaviour against closed-book QA and RAG.

## Notes on evaluation

The main automated evaluation uses an LLM-as-a-judge approach. Each generated answer is evaluated against the question, expected language, available context, reference answer, expected answer points, and content that should not be included.

The evaluation rubric includes:

* factual accuracy;
* grounding;
* clarity;
* usefulness;
* language correctness;
* scope handling;
* overall quality.

The LLM-as-a-judge evaluation provides a scalable first evaluation layer. Manual annotation files are included to support validation of the automated evaluation results.

## Notes on reproducibility

The original experiments were developed in Google Colab. Runtime may vary depending on GPU availability, model size, quantisation settings, and Hugging Face download speed.

For reproducible results, keep the following fixed:

* retrieval split random seed;
* benchmark dataset version;
* model IDs;
* quantisation settings;
* routing setup;
* generation parameters;
* prompt templates;
* retrieval thresholds;
* evaluation rubric.

## Important limitations

The datasets in this repository are constructed benchmark datasets, not real user conversations. They are intended for controlled prototype evaluation and should not be interpreted as a complete representation of real-world chatbot performance.

The current implementation focuses on English and Dutch. Support for additional languages would require additional translated or language-specific benchmark data, language detection validation, and retrieval testing.

The retrieval setup uses TF-IDF as a transparent baseline. More advanced retrieval methods, such as dense embeddings, hybrid retrieval, reranking, or document-level RAG over larger BeNext source materials, were not included in the current prototype.

The LLM-as-a-judge evaluation provides a scalable first evaluation layer, but automated scores may contain bias. Manual annotation files are included to support validation of the automated evaluation.

The experimental setup was constrained by available time and computational resources. The models, prompts, retrieval thresholds, decoding parameters, and retrieval methods were not exhaustively optimised. The results should therefore be interpreted as a controlled prototype comparison of feasible local approaches, not as proof that each approach was tested under its optimal configuration.

## Thesis context

This repository accompanies a thesis project on multilingual QA for energy-support chatbots. The goal is to compare whether closed-book QA, retrieval-based QA, or RAG is most suitable for answering residential energy questions in a multilingual setting, while considering answer quality, grounding, language correctness, latency, and practical feasibility.

## Author

Jakub Marcinek
Master Thesis Project
Applied Data Science, Utrecht University

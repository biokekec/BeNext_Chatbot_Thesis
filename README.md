# Multilingual QA for Energy Support

This repository contains the code and initial constructed datasets used for a thesis project on multilingual question answering for residential energy support. The project compares different chatbot approaches for answering user questions about household energy usage, energy monitoring, solar production, heat pumps, ventilation, and unusual energy patterns.

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
├── requirements.txt
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
* generating answers with selected open-source language models;
* saving model outputs and metadata for later evaluation.

This file should be treated as the main experimental notebook.

### `final_pipeline.py`

This file contains the cleaned final version of the pipeline with the final model and approach choices.

It is intended to provide a more reproducible version of the notebook workflow for further use, without unnecessary notebook outputs, manual display calls, or Colab-specific experimentation.

## Data

The `data/` folder contains the initial constructed benchmark datasets.

### `in_scope_QA.csv`

Contains constructed in-scope QA pairs related to residential energy support. Each row includes an English question, Dutch question, system-side context, reference answers, expected answer points, and content that should not be included in the chatbot answer.

### `out_of_scope_QA.csv`

Contains constructed out-of-scope examples. These are used to test whether the chatbot correctly refuses questions outside the supported domain, such as unrelated general questions, legal advice, device repair instructions, or other unsupported requests.

### `QA_guidelines.csv`

Contains answer guidelines used in the chatbot prompts. These guidelines define the desired answer style, scope boundaries, and safety constraints.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Some experiments require access to Hugging Face models. If a model requires authentication, add your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

Do not commit tokens, `.env` files, or private credentials to the repository.

## Notes on reproducibility

The original experiments were developed in Google Colab. Runtime may vary depending on GPU availability, model size, quantisation settings, and Hugging Face download speed.

For reproducible results, keep the following fixed:

* retrieval split random seed;
* model IDs;
* generation parameters;
* prompt templates;
* evaluation rubric;
* benchmark dataset version.

## Important limitations

The datasets in this repository are constructed benchmark datasets, not real user conversations. They are intended for controlled prototype evaluation and should not be interpreted as a complete representation of real-world chatbot performance.

The current implementation focuses on English and Dutch. Support for additional languages would require additional translated or language-specific benchmark data, validation, and retrieval testing.

## Thesis context

This repository accompanies a thesis project on multilingual QA for energy-support chatbots. The goal is to compare whether closed-book QA, retrieval-based QA, or RAG is most suitable for answering residential energy questions in a multilingual setting, while considering answer quality, grounding, language correctness, latency, and practical feasibility.

## Author

Jakub Marcinek
Master Thesis Project
Applied Data Science, Utrecht University
```

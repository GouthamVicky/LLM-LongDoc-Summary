## Research Paper Summarization

## Problem Statement

Develop a customized LLM model that can generate a summary of a given document.

## Proposed Solution 

- Preprocess the article's context
- Generate an extractive summary using sentence-transformer
- Generate a prompt dataset
- Fine-tune the [Falcon Model (1B)](tiiuae/falcon-rw-1b)
- Evaluate the Model output
- Serve the finetuned LLM model using [VLLM](https://github.com/vllm-project/vllm)
- Containerize the inference pipeline and deploy the API’s endpoint with [Streamlit](https://streamlit.io/) integrated UI

## Arxiv dataset for summarization

Dataset for summarization of long documents.\
Huggingface Link - [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization)

### Data Fields

- `id`: paper id
- `article`: a string containing the body of the paper
- `abstract`: a string containing the abstract of the paper


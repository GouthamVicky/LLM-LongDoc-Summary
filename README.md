##Research Paper Summarization

## Problem Statement

Develop a customized LLM model that can generate a summary of a given document.

## Arxiv dataset for summarization

Dataset for summarization of long documents.\
Adapted from this [repo](https://github.com/armancohan/long-summarization).\
Note that original data are pre-tokenized so this dataset returns " ".join(text) and add "\n" for paragraphs. \
This dataset is compatible with the [`run_summarization.py`](https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization) script from Transformers if you add this line to the `summarization_name_mapping` variable:
```python
"ccdv/arxiv-summarization": ("article", "abstract")
```

### Data Fields

- `id`: paper id
- `article`: a string containing the body of the paper
- `abstract`: a string containing the abstract of the paper


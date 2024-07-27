# RAG on rel-movielens1m

Performing classification and regression tasks using LLM and RAG on the rel-movielens1m dataset.

- Accuracy: MAE for regression; Micro-F1 and Macro-F1 for classification.
- Efficiency: Runtime, measured in seconds.
- Economy: Money, measured in dollars.

## Dependencies
### Environment
- Please refer to the `requirements.txt` file and follow the steps below to create a virtual environment and install the dependencies.
    - `conda create --name rllm python=3.9`
    - `conda activate rllm`
    - `pip install -r requirements.txt`

### LLM

- We use [Ollma](https://ollama.com) to locally run the llama3.1 model. You can follow these instructions to set up and run a local Ollama instance:
    - [Download](https://ollama.ai/download)
    - Fetch a model via `ollama pull llama3.1`

### LM

- We recommend a light BERT-like model  all-MiniLM-L6-v2 to make sentence embedding, which can be obtained from [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Please put the model into the `./resources/models/` directory after downloading.

### Docs for RAG
- Download [documents](https://jbox.sjtu.edu.cn/l/d13rps) and put them into the `./resources/datasets/wikidocs_wiki` directory.

## Classification
- Using `prompt` parameter to select different methods.
```shell
# Using title only：
python clf_rel-movielens1m.py --prompt title

# Using all information without genre: 
python clf_rel-movielens1m.py --prompt basic

# Using all infomation:
python clf_rel-movielens1m.py --prompt all

# Using Retrieval-Augmented Generation(RAG)
python clf_rel-movielens1m.py --prompt rag

# Using Chain-of-Thought(CoT)
python clf_rel-movielens1m.py --prompt basic_cot

# Using CoT together with RAG
python clf_rel-movielens1m.py --prompt rag_cot

# Using In-context Learning(ICL)
python clf_rel-movielens1m.py --prompt icl

# Using ICL together with RAG
python clf_rel-movielens1m.py --prompt rag_icl
```

- Using `dataset` parameter to select different datasets
  - a `training set` of 388 samples
  - a `test set` of 3107 samples


```shell
python clf_rel-movielens1m.py --dataset train
python clf_rel-movielens1m.py --dataset test
```


## Regression

```shell
python regression/rel-movielens1m_reg.py --prompt rag
python regression/rel-movielens1m_reg.py --prompt zero_shot
python regression/rel-movielens1m_reg.py --prompt compress
```
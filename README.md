# rLLM

**rLLM** (relation LLM) focuses on LLM-based relational data mining, prioritizing: Accuracy, Efficiency, and Economy.

- Accuracy: MAE for regression; Micro-F1 and Macro-F1 for classification.
- Efficiency: Runtime, measured in seconds.
- Economy: Money, measured in dollars.

## Dependencies

- pytorch	2.1.2 # conda install, not pip install
- scikit-learn	1.4.0
- llama_cpp_python	0.2.52
- langchain	0.1.8
- langchain-community	0.0.21
- langchain-experimental	0.0.52
- tiktoken	0.6.0
- sentence-transformers	2.3.1
- numpy	1.26.4
- pandas	2.1.4
- wikipedia
- sentence-transformers
- unstructured
- qdrant-client
- langchain_chroma
## LLM models

- We recommmend 4-bit quantized Gemma 2b model, which can be Downloaded from the SJTU cloud storage or [HuggingFace](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF/blob/main/gemma-2b-it-q4_k_m.gguf).
- Please put the model into the `./resources/model/` directory after downloading.

## LM Model

- We recommend a light BERT-like model  all-MiniLM-L6-v2 to make sentence embedding, which can be obtained from the SJTU cloud storage, or directly from [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Please put the model into the `./resources/model/` directory after downloading.

## Classification
- Using `title` only: 
```
python classification/rel-movielens1m_clf.py --prompt title
```
- Using all infomation:
```
python classification/rel-movielens1m_clf.py --prompt all
```

- Using RAG（TODO）

```
python classification/rel-movielens1m_clf.py --prompt rag
```

- You can choose different datasets, with a training set of 300 and a test set of 3000.

```
python classification/rel-movielens1m_clf.py --dataset train

python classification/rel-movielens1m_clf.py --dataset test
```

### RAG 初步方案
- 信息来源
    - 数据集 movie.csv 生成对应txt文件
    - wikipedia检索movie title，保存前两条结果为txt文件
- 下载 https://jbox.sjtu.edu.cn/l/81jgtG，解压到 `./resources/datasets/wikidocs/`
    - 只包含wiki内容 https://jbox.sjtu.edu.cn/l/d13rps
    - 只包含movie.csv生成的文档 https://jbox.sjtu.edu.cn/l/51MgbZ
- 使用 `sentence_transformer/all-MiniLM-L6-v2` 获取embedding
- 加载txt，进行切分和嵌入，向量化存储
- 对每部影片，查找相关文本，一起填入prompt_template，输入LLM，获得分类结果


## Regression

```
python regression/rel-movielens1m_reg.py
```
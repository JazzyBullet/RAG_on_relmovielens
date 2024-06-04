python classification/rel-movielens1m_clf.py --prompt title > ./results/classification/title.txt # baseline (w/o movie information)
python classification/rel-movielens1m_clf.py --prompt basic > ./results/classification/basic.txt # baseline (w/ movie information)
python classification/rel-movielens1m_clf.py --prompt basic_cot > ./results/classification/basic_cot.txt # baseline (w/ movie information) + CoT 
python classification/rel-movielens1m_clf.py --prompt all > ./results/classification/all.txt # SOTA
python classification/rel-movielens1m_clf.py --prompt rag > ./results/classification/rag.txt # RAG
python classification/rel-movielens1m_clf.py --prompt rag_cot > ./results/classification/rag_cot.txt # RAG + CoT
# all > RAG > RAG + CoT > basic_cot > basic > title
# all：全部信息
# basic：全部信息去掉genre
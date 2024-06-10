# -*- coding: ascii -*-

# Naive llm pipeline for regression task in rel-movielens1M
# Paper:
# MAE: 1.143
# Runtime: 26737s (on a single 6G GPU (Not applicable in practice))
# Cost: $6.9917
# Description: Use random 5 history rating to predict given user-movie rating by llm.
# Usage: python rel-movielens1m_reg.py

# Append rllm to search path
import sys
sys.path.append("../")
import time
import argparse
import pandas as pd
from tqdm import tqdm

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import BaseOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
import langchain
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rllm_RAG.utils import mae, get_llm_chat_cost, replace_punctuation_and_spaces, format_docs

##### Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', choices=['zero_shot', 'rag','compress'], 
                    default='zero_shot', help='Choose prompt type.')
args = parser.parse_args()


##### Start time
time_start = time.time()

##### Global variables
total_cost = 0
train_path = "./resources/datasets/rel-movielens1m/regression/ratings/train.csv"
movie_path = "./resources/datasets/rel-movielens1m/regression/movies.csv"
test_path = "./resources/datasets/rel-movielens1m/regression/ratings/test.csv"
llm_model_path = "./resources/model/gemma-2b-it-q4_k_m.gguf"

##### 1. Construct LLM chain
# Load model
llm = LlamaCpp(
    model_path=llm_model_path,
    streaming=False,
    n_gpu_layers=33,
    verbose=False,
    temperature=0.2,
    n_ctx=1024,
    stop=["Q", "\n", " "],
)

# Output parser
output_parser = StrOutputParser()

# Construct prompt
prompt_zero_shot_regression = """Q: Given a user's past movie ratings in the format: Title, Genres, Rating
Ratings range from 1.0 to 5.0.

{history_ratings}

The candidate movie is {candidate}. What's the rating that the user will give? 
Give a single number as rating without saying anything else.
A: """

prompt_rag_regression= """Q: Given a user's past movie ratings in the format: Title, Genres, Rating
Ratings range from 1.0 to 5.0.

{history_ratings}

The candidate movie is {candidate}.I have a movie description: {movie_info}. What's the rating that the user will give? 
Give a single number as rating without saying anything else.
A: """

prompt_compress="""Q: Now I have a movie description: {movie_info} summarize it.
A:"""


prompt_zero_shot_template = PromptTemplate(
    input_variables=["movie_name", "candidate"], template=prompt_zero_shot_regression)
prompt_rag_regression_template = PromptTemplate(
    input_variables=["movie_name", "candidate","movie_info"], template=prompt_rag_regression)
prompt_compress_template = PromptTemplate(
    input_variables=["movie_info"],
    template=prompt_compress
)
# Construct chain
if args.prompt=="zero_shot":
    prompt_template = prompt_zero_shot_template
    chain = prompt_zero_shot_template | llm | output_parser
elif args.prompt=="rag":
    prompt_template = prompt_rag_regression_template
    chain = prompt_rag_regression_template | llm | output_parser
elif args.prompt=="compress":
    prompt_template = prompt_rag_regression_template
    chain = prompt_rag_regression_template | llm | output_parser
    compress_chain = prompt_compress_template | llm | output_parser

##### 2. llm prediction
# Load files
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)
movie_data = pd.read_csv(movie_path)


# Prediction
def FindMovieDetail(movie_data: pd.DataFrame, movie_id: int) -> str:
    # Find MID and Genres
    movie_info = movie_data[movie_data["MovielensID"] == movie_id]
    movie_name = movie_info["Title"].values[0]
    genres = movie_info["Genre"].values[0]

    return f"{movie_name}, {genres}"

def FindMovieTitle(movie_data: pd.DataFrame, movie_id: int) -> str:
    # Find MID and Genres
    movie_info = movie_data[movie_data["MovielensID"] == movie_id]
    movie_name = movie_info["Title"].values[0]

    return movie_name


predict_ratings = []
if args.prompt=="zero_shot":
    # Get each UID and MID
    for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
        uid = row["UserID"]
        movie_id = row["MovieID"]

        # Find movie infomation
        movie_details = FindMovieDetail(movie_data, movie_id)

        # Find 5 random user history ratings
        user_ratings = train_data[train_data["UserID"] == uid].sample(n=5, random_state=42)
        history_movie_details_list = []

        # Get each MovieName and Genres
        for index, row in user_ratings.iterrows():
            movie_id = row["MovieID"]
            rating = row["Rating"]

            # Find history details
            history_movie_details = FindMovieDetail(movie_data, movie_id)
            history_movie_details = history_movie_details + f", {rating}"

            # Append history to list
            history_movie_details_list.append(history_movie_details)

        # use `\n` to concat
        history_movie_details_all = "\n".join(history_movie_details_list)

        total_cost = total_cost + get_llm_chat_cost(
            prompt_template.invoke(
                {"history_ratings": history_movie_details_all, "candidate": movie_details}
            ).text, 'input'
        )

        pred = chain.invoke(
            {"history_ratings": history_movie_details_all, "candidate": movie_details}
        )
        predict_ratings.append(float(pred))

        total_cost = total_cost + get_llm_chat_cost(pred, 'output')

elif args.prompt=="rag":
    # for sentence embedding
    embedding_model = HuggingFaceEmbeddings(model_name = "./resources/model/all-MiniLM-L6-v2")
    # Get each UID and MID
    for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
        uid = row["UserID"]
        movie_id = row["MovieID"]
        # Find movie infomation
        movie_details = FindMovieDetail(movie_data, movie_id)
        movie_title = FindMovieTitle(movie_data, movie_id)
        # load relevant documents
        # for network reason, we download wiki pages relating to relmovielens-1m dataset as txt files
        # and directly load them locally instead of load webpage again
        loader = DirectoryLoader("./resources/datasets/wikidocs/", glob="{}*.txt".format(replace_punctuation_and_spaces(movie_title)), loader_cls=TextLoader, use_multithreading=True)
        docs = loader.load()

        # split docs for embedding
        # chunk_size = 200, chunk_overlap = 0
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 0, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

        # convert to vector and store
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

        question = "what is the ratings of film or movie named '{}'".format(movie_title)
        movie_info=""
        for doc in retriever.invoke(question):
            movie_info+=doc.page_content+'\n'

        # Find 5 random user history ratings
        user_ratings = train_data[train_data["UserID"] == uid].sample(n=5, random_state=42)
        history_movie_details_list = []

        # Get each MovieName and Genres
        for index, row in user_ratings.iterrows():
            movie_id = row["MovieID"]
            rating = row["Rating"]

            # Find history details
            history_movie_details = FindMovieDetail(movie_data, movie_id)
            history_movie_details = history_movie_details + f", {rating}"

            # Append history to list
            history_movie_details_list.append(history_movie_details)

        # use `\n` to concat
        history_movie_details_all = "\n".join(history_movie_details_list)
        
        total_cost = total_cost + get_llm_chat_cost(
            prompt_template.invoke(
                {"history_ratings": history_movie_details_all, "candidate": movie_details, "movie_info": movie_info}
            ).text, 'input'
        )
        

        pred = chain.invoke(
            {"history_ratings": history_movie_details_all, "candidate": movie_details, "movie_info": movie_info}
        )
        predict_ratings.append(float(pred))

        total_cost = total_cost + get_llm_chat_cost(pred, 'output')
        
elif args.prompt=="compress":
    # for sentence embedding
    embedding_model = HuggingFaceEmbeddings(model_name = "./resources/model/all-MiniLM-L6-v2")
    # Get each UID and MID
    for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
        uid = row["UserID"]
        movie_id = row["MovieID"]
        # Find movie infomation
        movie_details = FindMovieDetail(movie_data, movie_id)
        movie_title = FindMovieTitle(movie_data, movie_id)
        # load relevant documents
        # for network reason, we download wiki pages relating to relmovielens-1m dataset as txt files
        # and directly load them locally instead of load webpage again
        loader = DirectoryLoader("./resources/datasets/wikidocs/", glob="{}*.txt".format(replace_punctuation_and_spaces(movie_title)), loader_cls=TextLoader, use_multithreading=True)
        docs = loader.load()

        # split docs for embedding
        # chunk_size = 200, chunk_overlap = 0
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 0, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

        # convert to vector and store
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

        question = "what is the ratings of film or movie named '{}'".format(movie_title)
        movie_info=""
        for doc in retriever.invoke(question):
            movie_info+=doc.page_content+'\n'
        movie_summary = compress_chain.invoke({"movie_info": movie_info})
        

        # Find 5 random user history ratings
        user_ratings = train_data[train_data["UserID"] == uid].sample(n=5, random_state=42)
        history_movie_details_list = []

        # Get each MovieName and Genres
        for index, row in user_ratings.iterrows():
            movie_id = row["MovieID"]
            rating = row["Rating"]

            # Find history details
            history_movie_details = FindMovieDetail(movie_data, movie_id)
            history_movie_details = history_movie_details + f", {rating}"

            # Append history to list
            history_movie_details_list.append(history_movie_details)

        # use `\n` to concat
        history_movie_details_all = "\n".join(history_movie_details_list)
        
        total_cost = total_cost + get_llm_chat_cost(
            prompt_template.invoke(
                {"history_ratings": history_movie_details_all, "candidate": movie_details, "movie_info": movie_info}
            ).text, 'input'
        )
        

        pred = chain.invoke(
            {"history_ratings": history_movie_details_all, "candidate": movie_details, "movie_info": movie_info}
        )
        
        predict_ratings.append(float(pred))

        total_cost = total_cost + get_llm_chat_cost(pred, 'output')


##### 3. Calculate MAE
real_ratings = list(test_data["Rating"])
mae_loss = mae(real_ratings, predict_ratings)

##### End time
time_end = time.time()

print(mae_loss)
print(f"Total time: {time_end - time_start}s")
print(f"Total USD$: {total_cost}")

import sys

sys.path.append("../")
import time
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from rllm_RAG.utils import (
    macro_f1_score,
    micro_f1_score,
    get_llm_chat_cost,
    format_docs,
    replace_punctuation_and_spaces,
)


##### Parse argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    choices=["title", "basic", "all", "rag", "basic_cot", "rag_cot", "icl", "rag_icl"],
    default="title",
    help="Choose prompt type.",
)
parser.add_argument(
    "--dataset", choices=["train", "test"], default="train", help="Choose dataset type."
)
args = parser.parse_args()


##### Start time
time_start = time.time()


##### Global variables
total_cost = 0
train_path = "./resources/datasets/rel-movielens1m/classification/movies/train.csv"
test_path = "./resources/datasets/rel-movielens1m/classification/movies/test.csv"
llm_model_name = "llama3.1"


##### 1. Construct LLM chain
# Load model
llm = Ollama(
    model=llm_model_name,
    num_gpu=1,
    verbose=False,
    temperature=0.2,
    num_ctx=1024,
    stop=["\n"],
)

# set genres
GenreSet = set(
    (
        "Documentary",
        "Adventure",
        "Comedy",
        "Horror",
        "War",
        "Sci-Fi",
        "Drama",
        "Mystery",
        "Western",
        "Action",
        "Children's",
        "Musical",
        "Thriller",
        "Crime",
        "Film-Noir",
        "Romance",
        "Animation",
        "Fantasy",
    )
)


class ExtractOutputParser(BaseOutputParser):
    """Parse the output of LLM to a genre list"""

    def parse(self, text: str):
        """Parse the output of LLM call."""
        genre_list = []
        for genre in GenreSet:
            if genre in text:
                genre_list.append(genre)
        return genre_list


output_parser = ExtractOutputParser()

# Construct prompt
prompt_title = """Q: Now I have a movie name: {movie_name}. What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:'Documentary', 'Adventure', 'Comedy', 'Horror', 'War', 'Sci-Fi', 'Drama', 'Mystery', 'Western', 'Action', "Children's", 'Musical', 'Thriller', 'Crime', 'Film-Noir', 'Romance', 'Animation', 'Fantasy'
3. Don't saying anything else.
4. You must answer at least one genre, empty answer is not allowed.
A: 
"""

prompt_basic = """Q: Now I have a movie description: The movie titled '{Title}' is directed by {Director} and was released in {Year}. The main cast of this movie include {Cast}. It has a runtime of {Runtime} and languages used including {Languages}, with a Certificate rating of {Certificate}. The plot summary is as follows: {Plot} What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:"Documentary", "Adventure", "Comedy", "Horror", "War", "Sci-Fi", "Drama", "Mystery", "Western", "Action", "Children's", "Musical", "Thriller", "Crime", "Film-Noir", "Romance", "Animation", "Fantasy"
3. Don't saying anything else.
4. You must answer at least one genre, empty answer is not allowed.
A: 
"""

prompt_all = """Q: Now I have a movie description: The movie titled '{Title}' is directed by {Director} and was released in {Year}. The genre of this movie is {Genre}, with main cast including {Cast}. It has a runtime of {Runtime} and languages used including {Languages}, with a Certificate rating of {Certificate}. The plot summary is as follows: {Plot} What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:"Documentary", "Adventure", "Comedy", "Horror", "War", "Sci-Fi", "Drama", "Mystery", "Western", "Action", "Children's", "Musical", "Thriller", "Crime", "Film-Noir", "Romance", "Animation", "Fantasy"
3. Don't saying anything else.
4. You must answer at least one genre, empty answer is not allowed.
A: 
"""

prompt_rag = """Q: Now I have a movie description: {movie_info} What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:"Documentary", "Adventure", "Comedy", "Horror", "War", "Sci-Fi", "Drama", "Mystery", "Western", "Action", "Children's", "Musical", "Thriller", "Crime", "Film-Noir", "Romance", "Animation", "Fantasy"
3. Don't saying anything else.
4. You must answer at least one genre, empty answer is not allowed.
A: 
"""

prompt_basic_cot = """Q: Now I have a movie description: The movie titled '{Title}' is directed by {Director} and was released in {Year}. The main cast of this movie include {Cast}. It has a runtime of {Runtime} and languages used including {Languages}, with a Certificate rating of {Certificate}. The plot summary is as follows: {Plot} What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n. Explain why the movie belongs to these genres.
2. The answer must only be chosen from followings:"Documentary", "Adventure", "Comedy", "Horror", "War", "Sci-Fi", "Drama", "Mystery", "Western", "Action", "Children's", "Musical", "Thriller", "Crime", "Film-Noir", "Romance", "Animation", "Fantasy"
3. Don't saying anything else.
4. You must answer at least one genre, empty answer is not allowed.
A: 
"""

prompt_rag_cot = """Q: Now I have a movie description: {movie_info} What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n. Explain why the movie belongs to these genres.
2. The answer must only be chosen from followings:"Documentary", "Adventure", "Comedy", "Horror", "War", "Sci-Fi", "Drama", "Mystery", "Western", "Action", "Children's", "Musical", "Thriller", "Crime", "Film-Noir", "Romance", "Animation", "Fantasy"
3. Don't saying anything else.
4. You must answer at least one genre, empty answer is not allowed.
A: 
"""

prompt_icl = """
The movie '{Title}': {Plot}. What's the genres it may belong to? 
{Title}:: {Genre}
"""

with open("./resources/icl.txt", "r+", encoding="utf-8") as f:
    icl = f.read()

prompt_rag_icl = (
    icl
    + """
Q: Now I have a movie description: {movie_info} What's the genres it may belong to? 
Note: 
1. Give the answer as following format:
movie_name:: genre_1, genre_2..., genre_n
2. The answer must only be chosen from followings:"Documentary", "Adventure", "Comedy", "Horror", "War", "Sci-Fi", "Drama", "Mystery", "Western", "Action", "Children's", "Musical", "Thriller", "Crime", "Film-Noir", "Romance", "Animation", "Fantasy"
3. Don't saying anything else.
4. You must answer at least one genre, empty answer is not allowed.
A: 
"""
)

prompt_title_template = PromptTemplate(
    input_variables=["movie_name"], template=prompt_title
)

prompt_basic_template = PromptTemplate(
    input_variables=[
        "Title",
        "Director",
        "Year",
        "Cast",
        "Runtime",
        "Languages",
        "Certificate",
        "Plot",
    ],
    template=prompt_basic,
)

prompt_all_template = PromptTemplate(
    input_variables=[
        "Title",
        "Director",
        "Year",
        "Genre",
        "Cast",
        "Runtime",
        "Languages",
        "Certificate",
        "Plot",
    ],
    template=prompt_all,
)

prompt_rag_template = PromptTemplate(
    input_variables=["movie_info"], template=prompt_rag
)

prompt_basic_cot_template = PromptTemplate(
    input_variables=[
        "Title",
        "Director",
        "Year",
        "Cast",
        "Runtime",
        "Languages",
        "Certificate",
        "Plot",
    ],
    template=prompt_basic_cot,
)

prompt_rag_cot_template = PromptTemplate(
    input_variables=["movie_info"], template=prompt_rag_cot
)

prompt_icl_template = PromptTemplate(
    input_variables=["Title", "Genre", "Plot"], template=prompt_icl
)

prompt_rag_icl_template = PromptTemplate(
    input_variables=["movie_info", "icl"], template=prompt_rag_icl
)

# Construct chain
if args.prompt == "title":
    chain = prompt_title_template | llm | output_parser
elif args.prompt == "rag":
    chain = prompt_rag_template | llm | output_parser
elif args.prompt == "all":
    chain = prompt_all_template | llm | output_parser
elif args.prompt == "icl":
    chain = prompt_icl_template | llm | output_parser
elif args.prompt == "rag_icl":
    chain = prompt_rag_icl_template | llm | output_parser
elif args.prompt == "basic":
    chain = prompt_basic_template | llm | output_parser
elif args.prompt == "basic_cot":
    chain = prompt_basic_cot_template | llm | output_parser
elif args.prompt == "rag_cot":
    chain = prompt_rag_cot_template | llm | output_parser


##### 2. LLM prediction
if args.dataset == "test":
    movie_df = pd.read_csv(test_path)
else:
    movie_df = pd.read_csv(train_path)

pred_genre_list = []
cnt_list = [2, 4, 9, 39, 40, 77, 184, 218, 230]

if args.prompt == "title":
    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        total_cost = total_cost + get_llm_chat_cost(
            prompt_title_template.invoke({"movie_name": row["Title"]}).text, "input"
        )
        pred = chain.invoke({"movie_name": row["Title"]})
        pred_genre_list.append(pred)
        total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

elif args.prompt == "basic":
    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        total_cost = total_cost + get_llm_chat_cost(
            prompt_basic_template.invoke(
                {
                    "Title": row["Title"],
                    "Director": row["Director"],
                    "Year": row["Year"],
                    "Cast": row["Cast"],
                    "Runtime": row["Runtime"],
                    "Languages": row["Languages"],
                    "Certificate": row["Certificate"],
                    "Plot": row["Plot"],
                }
            ).text,
            "input",
        )

        pred = chain.invoke(
            {
                "Title": row["Title"],
                "Director": row["Director"],
                "Year": row["Year"],
                "Cast": row["Cast"],
                "Runtime": row["Runtime"],
                "Languages": row["Languages"],
                "Certificate": row["Certificate"],
                "Plot": row["Plot"],
            }
        )
        pred_genre_list.append(pred)
        total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

elif args.prompt == "all":
    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        total_cost = total_cost + get_llm_chat_cost(
            prompt_all_template.invoke(
                {
                    "Title": row["Title"],
                    "Director": row["Director"],
                    "Year": row["Year"],
                    "Genre": row["Genre"],
                    "Cast": row["Cast"],
                    "Runtime": row["Runtime"],
                    "Languages": row["Languages"],
                    "Certificate": row["Certificate"],
                    "Plot": row["Plot"],
                }
            ).text,
            "input",
        )

        pred = chain.invoke(
            {
                "Title": row["Title"],
                "Director": row["Director"],
                "Year": row["Year"],
                "Genre": row["Genre"],
                "Cast": row["Cast"],
                "Runtime": row["Runtime"],
                "Languages": row["Languages"],
                "Certificate": row["Certificate"],
                "Plot": row["Plot"],
            }
        )
        pred_genre_list.append(pred)
        total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

elif args.prompt == "rag":
    # for sentence embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="./resources/models/all-MiniLM-L6-v2"
    )

    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        # load relevant documents
        # for network reason, we download wiki pages relating to relmovielens-1m dataset as txt files
        # and directly load them locally instead of load webpage again
        loader = DirectoryLoader(
            "./resources/datasets/wikidocs_wiki/",
            glob="{}*.txt".format(replace_punctuation_and_spaces(row["Title"])),
            loader_cls=TextLoader,
            use_multithreading=True,
        )
        docs = loader.load()

        if len(docs) != 0:
            # split docs for embedding
            # chunk_size = 200, chunk_overlap = 0
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200, chunk_overlap=0, add_start_index=True
            )
            all_splits = text_splitter.split_documents(docs)

            # convert to vector and store
            vectorstore = Chroma.from_documents(
                documents=all_splits, embedding=embedding_model
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

            # construct RAG chain
            rag_chain = {
                "movie_info": retriever | format_docs,
                "question": RunnablePassthrough(),
            } | chain

            # utilize RAG for movie classification
            question = "what is the genre of film or movie named '{}'".format(
                row["Title"]
            )
            pred = rag_chain.invoke(question)
            pred_genre_list.append(list(set(pred)))

            # calculate the cost
            total_cost = total_cost + get_llm_chat_cost(
                prompt_rag_template.invoke(
                    {"movie_info": retriever.invoke(question)}
                ).text,
                "input",
            )
            total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

        else:
            # no related docs, using title only
            total_cost = total_cost + get_llm_chat_cost(
                prompt_rag_template.invoke({"movie_info": row["Title"]}).text, "input"
            )
            question = "what is the genre of film or movie named '{}'".format(
                row["Title"]
            )
            pred = chain.invoke(question)
            pred_genre_list.append(pred)
            total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

elif args.prompt == "basic_cot":
    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        total_cost = total_cost + get_llm_chat_cost(
            prompt_basic_cot_template.invoke(
                {
                    "Title": row["Title"],
                    "Director": row["Director"],
                    "Year": row["Year"],
                    "Cast": row["Cast"],
                    "Runtime": row["Runtime"],
                    "Languages": row["Languages"],
                    "Certificate": row["Certificate"],
                    "Plot": row["Plot"],
                }
            ).text,
            "input",
        )

        pred = chain.invoke(
            {
                "Title": row["Title"],
                "Director": row["Director"],
                "Year": row["Year"],
                "Cast": row["Cast"],
                "Runtime": row["Runtime"],
                "Languages": row["Languages"],
                "Certificate": row["Certificate"],
                "Plot": row["Plot"],
            }
        )
        pred_genre_list.append(pred)
        total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

elif args.prompt == "rag_cot":
    # for sentence embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="./resources/models/all-MiniLM-L6-v2"
    )

    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        # load relevant documents
        # for network reason, we download wiki pages relating to relmovielens-1m dataset as txt files
        # and directly load them locally instead of load webpage again
        loader = DirectoryLoader(
            "./resources/datasets/wikidocs_wiki/",
            glob="{}*.txt".format(replace_punctuation_and_spaces(row["Title"])),
            loader_cls=TextLoader,
            use_multithreading=True,
        )
        docs = loader.load()

        if len(docs) != 0:
            # split docs for embedding
            # chunk_size = 200, chunk_overlap = 0
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200, chunk_overlap=0, add_start_index=True
            )
            all_splits = text_splitter.split_documents(docs)

            # convert to vector and store
            vectorstore = Chroma.from_documents(
                documents=all_splits, embedding=embedding_model
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

            # construct RAG chain
            rag_chain = {
                "movie_info": retriever | format_docs,
                "question": RunnablePassthrough(),
            } | chain

            # utilize RAG for movie classification
            question = "what is the genre of film or movie named '{}'".format(
                row["Title"]
            )
            pred = rag_chain.invoke(question)
            # del loader, docs, text_splitter, all_splits, vectorstore, retriever, rag_chain
            pred_genre_list.append(list(set(pred)))

            # calculate the cost
            total_cost = total_cost + get_llm_chat_cost(
                prompt_rag_template.invoke(
                    {"movie_info": retriever.invoke(question)}
                ).text,
                "input",
            )
            total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

        else:
            # no related docs, using title only
            total_cost = total_cost + get_llm_chat_cost(
                prompt_rag_template.invoke({"movie_info": row["Title"]}).text, "input"
            )
            question = "what is the genre of film or movie named '{}'".format(
                row["Title"]
            )
            pred = chain.invoke(question)
            pred_genre_list.append(pred)
            total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

elif args.prompt == "icl":
    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        if index + 2 in cnt_list:
            Genre_format = ",".join(row["Genre"].split("|"))
            Q = prompt_icl_template.format(
                Title=row["Title"], Genre=Genre_format, Plot=row["Plot"]
            )
            print(Q)

elif args.prompt == "rag_icl":
    # for sentence embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="./resources/models/all-MiniLM-L6-v2"
    )

    for index, row in tqdm(
        movie_df.iterrows(), total=len(movie_df), desc="Processing Movies"
    ):
        # load relevant documents
        # for network reason, we download wiki pages relating to relmovielens-1m dataset as txt files
        # and directly load them locally instead of load webpage again
        loader = DirectoryLoader(
            "./resources/datasets/wikidocs_wiki/",
            glob="{}*.txt".format(replace_punctuation_and_spaces(row["Title"])),
            loader_cls=TextLoader,
            use_multithreading=True,
        )
        docs = loader.load()

        if len(docs) != 0:
            # split docs for embedding
            # chunk_size = 200, chunk_overlap = 0
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200, chunk_overlap=0, add_start_index=True
            )
            all_splits = text_splitter.split_documents(docs)

            # convert to vector and store
            vectorstore = Chroma.from_documents(
                documents=all_splits, embedding=embedding_model
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

            # construct RAG chain
            rag_chain = {
                "movie_info": retriever | format_docs,
                "question": RunnablePassthrough(),
            } | chain

            # utilize RAG for movie classification
            question = "what is the genre of film or movie named '{}'".format(
                row["Title"]
            )
            pred = rag_chain.invoke(question)
            # del loader, docs, text_splitter, all_splits, vectorstore, retriever, rag_chain
            pred_genre_list.append(list(set(pred)))

            # calculate the cost
            total_cost = total_cost + get_llm_chat_cost(
                prompt_rag_template.invoke(
                    {"movie_info": retriever.invoke(question)}
                ).text,
                "input",
            )
            total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")

        else:
            # no related docs, using title only
            total_cost = total_cost + get_llm_chat_cost(
                prompt_rag_template.invoke({"movie_info": row["Title"]}).text, "input"
            )
            question = "what is the genre of film or movie named '{}'".format(
                row["Title"]
            )
            pred = chain.invoke(question)
            pred_genre_list.append(pred)
            total_cost = total_cost + get_llm_chat_cost(",".join(pred), "output")


##### 3. Calculate macro f1 score
# Get all genres
movie_genres = movie_df["Genre"].str.split("|")
all_genres = list(set([genre for genres in movie_genres for genre in genres]))
all_genres = [genre.lower() for genre in all_genres]
movie_genres = [[s.lower() for s in sublist] for sublist in movie_genres]
pred_genre_list = [[s.lower() for s in sublist] for sublist in pred_genre_list]

mlb = MultiLabelBinarizer(classes=all_genres)
real_genres_matrix = mlb.fit_transform(movie_genres)
pred_genres_matrix = mlb.fit_transform(pred_genre_list)
macro_f1 = macro_f1_score(real_genres_matrix, pred_genres_matrix)
micro_f1 = micro_f1_score(real_genres_matrix, pred_genres_matrix)


##### End time
time_end = time.time()


##### Results
print(f"macro_f1: {macro_f1}")
print(f"micro_f1: {micro_f1}")
print(f"Total time: {time_end - time_start}s")
print(f"Total USD$: {total_cost}")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle


def scripts_rework(path, character):
    """FOR GENERARTIVE MODEL TRAINING!!!
    this functions split scripts for question, answer, context,
    picks up the character, augments data for generative model training
    and saves data in pickle format"""

    df = pd.read_csv(path)

    # split data for scenes
    count = 0
    df["scene_count"] = ""
    for index, row in df.iterrows():
        if index == 0:
            df.iloc[index]["scene_count"] = count
        elif row["person_scene"] == "Scene":
            count += 1
            df.iloc[index]["scene_count"] = count
        else:
            df.iloc[index]["scene_count"] = count

    df = df.dropna().reset_index()

    # rework scripts to filer by caracter utterances and related context
    scripts = pd.DataFrame()
    for index, row in df.iterrows():
        if (row["person_scene"] == character) & (
            df.iloc[index - 1]["person_scene"] != "Scene"
        ):
            context = []

            for i in reversed(range(2, 6)):
                if (df.iloc[index - i]["person_scene"] != "Scene") & (index - i >= 0):
                    context.append(df.iloc[index - i]["dialogue"])
                else:
                    break

            for j in range(len(context)):
                new_row = {
                    "answer": row["dialogue"],
                    "question": df.iloc[index - 1]["dialogue"],
                    "context": context[j:],
                }
                scripts = pd.concat([scripts, pd.DataFrame([new_row])])
            new_row = {
                "answer": row["dialogue"],
                "question": df.iloc[index - 1]["dialogue"],
                "context": [],
            }
            scripts = pd.concat([scripts, pd.DataFrame([new_row])])

        elif (row["person_scene"] == character) & (
            df.iloc[index - 1]["person_scene"] == "Scene"
        ):
            context = []
            new_row = {"answer": row["dialogue"], "question": "", "context": context}
            scripts = pd.concat([scripts, pd.DataFrame([new_row])])
    # load reworked data to pkl
    scripts = scripts[scripts["question"] != ""]
    scripts["context"] = scripts["context"].apply(lambda x: "".join(x))
    scripts = scripts.reset_index(drop=True)
    scripts.to_pickle("data/scripts_reworked.pkl")


# ===================================================
def scripts_rework_ranking(path, character):
    """FOR RAG RETRIEVAL !!!!
    this functions split scripts for queation, answer, context,
    picks up the cahracter and saves data in pickle format"""

    df = pd.read_csv(path)

    # split data for scenes
    count = 0
    df["scene_count"] = ""
    for index, row in df.iterrows():
        if index == 0:
            df.iloc[index]["scene_count"] = count
        elif row["person_scene"] == "Scene":
            count += 1
            df.iloc[index]["scene_count"] = count
        else:
            df.iloc[index]["scene_count"] = count

    df = df.dropna().reset_index()

    # rework scripts to filer by caracter utterances and related context
    scripts = pd.DataFrame()
    for index, row in df.iterrows():
        if (row["person_scene"] == character) & (
            df.iloc[index - 1]["person_scene"] != "Scene"
        ):
            context = []
            for i in reversed(range(2, 5)):
                if (df.iloc[index - i]["person_scene"] != "Scene") & (index - i >= 0):
                    context.append(df.iloc[index - i]["dialogue"])
                else:
                    break
            new_row = {
                "answer": row["dialogue"],
                "question": df.iloc[index - 1]["dialogue"],
                "context": context,
            }

            scripts = pd.concat([scripts, pd.DataFrame([new_row])])

        elif (row["person_scene"] == character) & (
            df.iloc[index - 1]["person_scene"] == "Scene"
        ):
            context = []
            new_row = {"answer": row["dialogue"], "question": "", "context": context}
            scripts = pd.concat([scripts, pd.DataFrame([new_row])])
    # load reworked data to pkl
    scripts = scripts[scripts["question"] != ""]
    scripts = scripts.reset_index(drop=True)
    scripts.to_pickle("data/scripts.pkl")


# ===================================================
def encode(texts, model, contexts=None, do_norm=True):
    """function to encode texts for cosine similarity search"""

    question_vectors = model.encode(texts)
    if type(contexts) is list:
        context_vectors = model.encode("".join(contexts))
    else:
        context_vectors = model.encode(contexts)

    return np.concatenate(
        [
            np.asarray(context_vectors),
            np.asarray(question_vectors),
        ],
        axis=-1,
    )


def encode_rag(texts, model, contexts=None, do_norm=True):
    """function to encode texts for cosine similarity search"""

    question_vectors = model.encode(texts)
    context_vectors = model.encode("".join(contexts))

    return np.concatenate(
        [
            np.asarray(context_vectors),
            np.asarray(question_vectors),
        ],
        axis=-1,
    )


# ===================================================
def encode_df_save(model):
    """FOR RAG RETRIEVAL DATABASE
    this functions vectorizes reworked scripts and loads them to
    pickle file to be used as retrieval base for ranking script"""

    scripts_reopened = pd.read_pickle("data/scripts.pkl")
    vect_data = []
    for index, row in scripts_reopened.iterrows():
        if type(row["context"]) is list:
            vect = encode(
                texts=row["question"],
                model=model,
                contexts="".join(row["context"]),
            )
            vect_data.append(vect)
        else:
            vect = encode(
                texts=row["question"],
                model=model,
                contexts=row["context"],
            )
            vect_data.append(vect)
    with open("data/scripts_vectors.pkl", "wb") as f:
        pickle.dump(vect_data, f)


# ===================================================
def cosine_sim(answer_true_vectros, answer_generated_vectors) -> list:
    """FOR MODEL EVALUATION!!!!
    returns list of tuples with similarity score"""

    data_emb = sparse.csr_matrix(answer_true_vectros)
    query_emb = sparse.csr_matrix(answer_generated_vectors)
    similarity = cosine_similarity(query_emb, data_emb).flatten()
    return similarity[0]


# ===================================================
def cosine_sim_rag(data_vectors, query_vectors) -> list:
    """FOR RAG RETRIEVAL RANKS!!!
    returns list of tuples with similarity score and
    script index in initial dataframe"""

    data_emb = sparse.csr_matrix(data_vectors)
    query_emb = sparse.csr_matrix(query_vectors)
    similarity = cosine_similarity(query_emb, data_emb).flatten()
    ind = np.argwhere(similarity)
    match = sorted(zip(similarity, ind.tolist()), reverse=True)

    return match


# ===================================================
def generate_response(
    model,
    tokenizer,
    question,
    context,
    top_p,
    temperature,
    rag_answer="",
):

    combined = (
        "context:" + rag_answer +
        "".join(context) + "</s>" +
        "question: " + question
    )
    input_ids = tokenizer.encode(combined, return_tensors="pt")
    sample_output = model.generate(
        input_ids,
        do_sample=True,
        max_length=1000,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=2.0,
        top_k=50,
        no_repeat_ngram_size=4,
        # early_stopping=True,
        # min_length=10,
    )

    out = tokenizer.decode(sample_output[0][1:], skip_special_tokens=True)
    if "</s>" in out:
        out = out[: out.find("</s>")].strip()

    return out


# ===================================================
def top_candidates(score_lst_sorted, initial_data, top=1):
    """this functions receives results of the cousine similarity ranking and
    returns top items' scores and their indices"""

    scores = [item[0] for item in score_lst_sorted]
    candidates_indexes = [item[1][0] for item in score_lst_sorted]
    return scores[0:top], candidates_indexes[0:top]

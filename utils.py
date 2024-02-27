import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def scripts_rework(path, character):
    """this functions split scripts for queation, answer, context,
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


def encode(texts, model, contexts=None, do_norm=True):
    """function to encode texts for cosine similarity search"""

    question_vectors = model.encode(texts)
    context_vectors = model.encode(contexts)

    return np.concatenate(
        [
            np.asarray(context_vectors),
            np.asarray(question_vectors),
        ],
        axis=-1,
    )


def cosine_sim(answer_true_vectros, answer_generated_vectors) -> list:
    """returns list of tuples with similarity score and
    script index in initial dataframe"""

    data_emb = sparse.csr_matrix(answer_true_vectros)
    query_emb = sparse.csr_matrix(answer_generated_vectors)
    similarity = cosine_similarity(query_emb, data_emb).flatten()
    return similarity[0]


def reranking_score(question, context, answer, model):
    """this function applies trained bert classifier to identified candidates
      and returns their updated rank"""
    combined_text = ''.join(context) + " [SEP] " + question + " [SEP] " + answer

    prediction = model(combined_text)
    if prediction[0]["label"] == "LABEL_0":
        reranked_score = prediction[0]["score"]
        return reranked_score
    else:
        return 0


def generate_response(model, tokenizer, question, context, top_p, temperature):
    combined = "context: " + ''.join(context) + "</s>" + "question: " + question
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


def reranking_output(beam_answers_lst, question, context, model):
    """this function applies trained bert classifier to generated candidates and
    returns the one with max rank"""
    score_lst = []
    for beam_answer in beam_answers_lst:
        reranker_score = reranking_score(
            question=question, context=context, answer=beam_answer, model=model
        )

        score_lst.append(reranker_score)
    return beam_answers_lst[score_lst.index(max(score_lst))]

import pandas as pd


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
    

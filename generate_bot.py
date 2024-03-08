from collections import deque
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from utils import generate_response
import pandas as pd
import pickle
from utils import encode_rag, cosine_sim_rag, top_candidates


class ChatBot:
    def __init__(self):
        self.conversation_history = deque([], maxlen=10)
        self.generative_model = None
        self.generative_tokenizer = None
        self.vect_data = []
        self.scripts = []
        self.ranking_model = None

    def load(self):
        """ "This method is called first to load all datasets and
        model used by the chat bot; all the data to be saved in
        tha data folder, models to be loaded from hugging face"""

        with open("data/scripts_vectors.pkl", "rb") as fp:
            self.vect_data = pickle.load(fp)
        self.scripts = pd.read_pickle("data/scripts.pkl")
        self.ranking_model = SentenceTransformer(
            "Shakhovak/chatbot_sentence-transformer"
        )
        self.generative_model = AutoModelForSeq2SeqLM.from_pretrained(
            "Shakhovak/flan-t5-base-sheldon-chat-v2"
        )
        self.generative_tokenizer = AutoTokenizer.from_pretrained(
            "Shakhovak/flan-t5-base-sheldon-chat-v2"
        )

    def generate_response(self, utterance):

        query_encoding = encode_rag(
            texts=utterance,
            model=self.ranking_model,
            contexts=self.conversation_history,
        )

        bot_cosine_scores = cosine_sim_rag(
            self.vect_data,
            query_encoding,
        )

        top_scores, top_indexes = top_candidates(
            bot_cosine_scores, initial_data=self.scripts
        )

        if top_scores[0] >= 0.89:
            for index in top_indexes:
                rag_answer = self.scripts.iloc[index]["answer"]

            answer = generate_response(
                model=self.generative_model,
                tokenizer=self.generative_tokenizer,
                question=utterance,
                context=self.conversation_history,
                top_p=0.9,
                temperature=0.95,
                rag_answer=rag_answer,
            )
        else:
            answer = generate_response(
                model=self.generative_model,
                tokenizer=self.generative_tokenizer,
                question=utterance,
                context=self.conversation_history,
                top_p=0.9,
                temperature=0.95,
            )

        self.conversation_history.append(utterance)
        self.conversation_history.append(answer)
        return answer


# katya = ChatBot()
# katya.load()
# print(katya.generate_response("What is he doing there?"))

from collections import deque
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import generate_response


class ChatBot:
    def __init__(self):
        self.conversation_history = deque([], maxlen=6)
        self.generative_model = None
        self.generative_tokenizer = None

    def load(self):
        self.generative_model = AutoModelForSeq2SeqLM.from_pretrained(
            "Shakhovak/flan-t5-base-sheldon-chat"
        )
        self.generative_tokenizer = AutoTokenizer.from_pretrained(
            "Shakhovak/flan-t5-base-sheldon-chat"
        )

    def generate_response(self, utterance):

        answer = generate_response(
            model=self.generative_model,
            tokenizer=self.generative_tokenizer,
            question=utterance,
            context=self.conversation_history,
            top_p=0.95,
            temperature=1,
        )
        self.conversation_history.append(utterance)
        self.conversation_history.append(answer)
        return answer


# katya = ChatBot()
# katya.load()
# print(katya.generate_response("What is he doing there?"))

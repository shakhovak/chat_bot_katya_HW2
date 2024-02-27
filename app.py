from flask import Flask, render_template, request
from generate_bot import ChatBot
import asyncio

app = Flask(__name__)
chatSheldon = ChatBot()
chatSheldon.load()

# this script is running flask application


@app.route("/")
def index():
    return render_template("chat.html")


async def sleep():
    await asyncio.sleep(1)
    return 1


@app.route("/get", methods=["GET", "POST"])
async def chat():
    msg = request.form["msg"]
    input = msg
    await asyncio.gather(sleep(), sleep())
    return get_Chat_response(input)


def get_Chat_response(text):
    answer = chatSheldon.generate_response(text)
    return answer


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

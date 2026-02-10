from flask import Flask, request, jsonify, render_template
from rag_core import ask_rag
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chatbot")
def chatbot():
    return render_template("chat_bot.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"reply": "Please ask a valid question."})

    answer = ask_rag(user_query)
    return jsonify({"reply": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


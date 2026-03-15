from flask import Flask, render_template, request, jsonify
from model import ai_therapist_pipeline  # import your pipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]
    emotion, reply = ai_therapist_pipeline(user_msg)
    return jsonify({"reply": reply, "emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)

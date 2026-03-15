from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os
from textblob import TextBlob
import random

# -----------------------------
# 1. Load Emotion Classifier (BERT)
# -----------------------------
emotion_model_path = "ai_therapist_emotion_classifier"
emotion_tokenizer = BertTokenizerFast.from_pretrained(emotion_model_path)
emotion_model = BertForSequenceClassification.from_pretrained(emotion_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model.to(device)

label_map = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# -----------------------------
# 2. Load Response Generator (FLAN-T5-large)
# -----------------------------
response_model_path = r"C:\Users\venka\Desktop\FLA-T5-Large"
response_tokenizer = T5Tokenizer.from_pretrained(response_model_path, local_files_only=True)
response_model = T5ForConditionalGeneration.from_pretrained(response_model_path, local_files_only=True)
response_model.to(device)

# -----------------------------
# 3. Emotion templates
# -----------------------------
templates = {
    "sadness": "I see that you're feeling sad. {user_text} It's okay to feel this way. Remember, you are not alone.",
    "anger": "I understand you are angry. {user_text} Let's try to calm down and think about what we can do next.",
    "fear": "Feeling afraid is natural. {user_text} Take deep breaths and remember you are safe.",
    "nervousness": "It's normal to feel nervous. {user_text} Focus on small steps and take it one thing at a time.",
    "joy": "That's wonderful! {user_text} Keep celebrating your happiness!",
    "neutral": "{user_text} Thank you for sharing your thoughts with me."
}

# -----------------------------
# 4. Conversation memory
# -----------------------------
class ConversationMemory:
    def __init__(self, max_turns=5):
        self.max_turns = max_turns  # number of previous exchanges to remember
        self.history = []  # list of tuples: (user_text, therapist_reply)

    def add_turn(self, user_text, therapist_reply):
        self.history.append((user_text, therapist_reply))
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context_prompt(self):
        prompt = ""
        for i, (user, reply) in enumerate(self.history):
            prompt += f"Previous conversation {i+1}:\nUser: {user}\nTherapist: {reply}\n\n"
        return prompt

conversation_memory = ConversationMemory(max_turns=5)

# -----------------------------
# 5. Helper Functions
# -----------------------------
def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    emotion_model.eval()
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred_id]

def compute_emotion_intensity(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    length = len(text.split())
    score = 0
    if polarity < -0.2:
        score += 1
    if length > 15:
        score += 1
    if "!" in text or "?" in text:
        score += 1
    if score <= 1:
        return "mild"
    elif score == 2:
        return "moderate"
    else:
        return "high"

def generate_therapist_reply(user_text, emotion_label, memory: ConversationMemory):
    intensity = compute_emotion_intensity(user_text)
    base_text = templates.get(emotion_label, user_text).replace("{user_text}", user_text)
    context_prompt = memory.get_context_prompt()

    # Dynamic generation parameters
    if intensity == "mild":
        max_tokens = random.randint(120, 160)
        temperature = round(random.uniform(0.6, 0.7), 2)
        top_p = round(random.uniform(0.85, 0.9), 2)
    elif intensity == "moderate":
        max_tokens = random.randint(160, 200)
        temperature = round(random.uniform(0.7, 0.8), 2)
        top_p = round(random.uniform(0.9, 0.95), 2)
    else:  # high intensity
        max_tokens = random.randint(200, 300)
        temperature = round(random.uniform(0.8, 0.9), 2)
        top_p = round(random.uniform(0.9, 0.97), 2)

    # Few-shot + context prompt
    prompt = f"""
You are a kind and empathetic AI therapist. You remember previous conversations. 
Follow these steps:
1. Acknowledge the user's emotion and intensity.
2. Offer supportive advice, coping strategies, and encouragement.
3. Keep the response empathetic and suitable for the emotion's intensity.
4. Use varied phrasing for naturalness.

{context_prompt}

User: "{base_text}"
Emotion: {emotion_label}
Intensity: {intensity}
Response:
"""

    inputs = response_tokenizer(prompt.strip(), return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = response_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=50,
        repetition_penalty=1.2
    )

    reply = response_tokenizer.decode(outputs[0], skip_special_tokens=True)
    memory.add_turn(user_text, reply)
    return reply

# -----------------------------
# 6. Unified Pipeline
# -----------------------------
def ai_therapist_pipeline(user_text):
    emotion = predict_emotion(user_text)
    response = generate_therapist_reply(user_text, emotion, conversation_memory)
    return emotion, response

# -----------------------------
# 7. Example Conversation
# -----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        detected_emotion, therapist_reply = ai_therapist_pipeline(user_input)
        print(f"Detected Emotion: {detected_emotion}")
        print(f"Therapist: {therapist_reply}\n")

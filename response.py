# -----------------------------
# AI Therapist: Response Model (FLAN-T5-base)
# Few-shot + deterministic decoding for consistent outputs
# -----------------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load FLAN-T5-base model
model_id = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

# Function to generate empathetic response
def generate_therapist_reply(user_text, emotion_label):
    """
    Generates a consistent, empathetic therapeutic response
    user_text: str - User's message
    emotion_label: str - Detected emotion from your model
    """
    
    # Few-shot examples for guidance
    prompt = f"""
Example 1:
User: "I feel anxious about tomorrow"
Emotion: nervousness
Response: "It's normal to feel anxious. Take deep breaths and remember you are capable of handling tomorrow."

Example 2:
User: "I feel like no one cares about me anymore"
Emotion: sadness
Response: "I'm really sorry you're feeling that way. Your feelings are valid, and you are not alone."

Example 3:
User: "I'm so excited about my new job!"
Emotion: joy
Response: "That's wonderful! It's great to hear how happy you are about this opportunity."

Now, given the following input, provide a kind, supportive, and empathetic response:

User: "{user_text}"
Emotion: {emotion_label}
Response:
"""
    
    inputs = tokenizer(prompt.strip(), return_tensors="pt", truncation=True)
    
    # Deterministic generation
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False  # disables randomness
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    user_input = "I feel like no one listens to me"
    detected_emotion = "sadness"  # from your emotion detection model

    reply = generate_therapist_reply(user_input, detected_emotion)
    print("Therapist:", reply)

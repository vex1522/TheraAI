from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

# Load tokenizer and trained model
model_path = "ai_therapist_emotion_classifier"  # Path to your saved model
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Emotion labels
label_map = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# Function to predict emotion
def predict_emotion(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to same device as model
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        return label_map[pred_id]
    except Exception as e:
        return f"Error: {e}"

# Example usage
if __name__ == "__main__":
    sample = "I think i have found love"  # Example input
    predicted_emotion = predict_emotion(sample)
    print(f"Predicted Emotion: {predicted_emotion}")

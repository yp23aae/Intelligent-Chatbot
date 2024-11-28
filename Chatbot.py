# Import required libraries
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from flask import Flask, request, jsonify
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Initialize Flask app
app = Flask(__name__)

# -------------------- Step 1: Load and Train Models --------------------
# Load and preprocess GoEmotions dataset
def load_and_tokenize_emotion_data():
    dataset = load_dataset("go_emotions")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_data(data):
        return tokenizer(data["text"], truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset, tokenizer


# Fine-tune BERT for emotion detection
def train_emotion_model():
    tokenized_dataset, tokenizer = load_and_tokenize_emotion_data()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)

    training_args = TrainingArguments(
        output_dir="./emotion_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    trainer.train()
    model.save_pretrained("emotion_model")
    tokenizer.save_pretrained("emotion_model")
    return model, tokenizer


# Load and preprocess EmpatheticDialogues dataset
def load_and_tokenize_response_data():
    dataset = load_dataset("empathetic_dialogues")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def preprocess_data(data):
        return tokenizer(data["utterance"], truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    return tokenized_dataset, tokenizer


# Fine-tune GPT-2 for empathetic responses
def train_response_model():
    tokenized_dataset, tokenizer = load_and_tokenize_response_data()
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir="./response_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()
    model.save_pretrained("response_model")
    tokenizer.save_pretrained("response_model")
    return model, tokenizer


# -------------------- Step 2: Load Pretrained Models --------------------
# Load trained models
try:
    emotion_model = AutoModelForSequenceClassification.from_pretrained("emotion_model")
    emotion_tokenizer = AutoTokenizer.from_pretrained("emotion_model")
    response_model = GPT2LMHeadModel.from_pretrained("response_model")
    response_tokenizer = GPT2Tokenizer.from_pretrained("response_model")
except:
    print("Training models...")
    emotion_model, emotion_tokenizer = train_emotion_model()
    response_model, response_tokenizer = train_response_model()

# -------------------- Step 3: Flask API --------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    # Step 1: Emotion Detection
    emotion_inputs = emotion_tokenizer(user_input, return_tensors="pt")
    emotion_outputs = emotion_model(**emotion_inputs)
    emotion = torch.argmax(emotion_outputs.logits, dim=1).item()

    # Map emotion ID to emotion label (GoEmotions)
    emotion_labels = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ]
    detected_emotion = emotion_labels[emotion]

    # Step 2: Generate Response
    prompt = f"Emotion: {detected_emotion}\nUser: {user_input}\nBot:"
    input_ids = response_tokenizer(prompt, return_tensors="pt").input_ids
    response_ids = response_model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = response_tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return jsonify({"response": response, "detected_emotion": detected_emotion})


# -------------------- Step 4: Run the Application --------------------
if __name__ == "__main__":
    app.run(debug=True)

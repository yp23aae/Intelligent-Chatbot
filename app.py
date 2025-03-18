from flask import Flask, request, jsonify,send_file
import pickle
import numpy as np

app = Flask(__name__)


with open('emotion_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


def detect_emotion(text):
   
    text_tfidf = vectorizer.transform([text])
   
    emotion_prediction = model.predict(text_tfidf)
   
    detected_emotions = [emotion_labels[i] for i, val in enumerate(emotion_prediction[0]) if val == 1]
    return detected_emotions


def generate_response(emotions):
    if not emotions:
        return "I'm not sure how you're feeling. Can you tell me more?"
    
    response = "I sense that you're feeling "
    if len(emotions) == 1:
        response += emotions[0] + "."
    else:
        response += ", ".join(emotions[:-1]) + ", and " + emotions[-1] + "."
    

    if 'joy' in emotions:
        response += " That's great to hear! 😊"
    elif 'sadness' in emotions:
        response += " I'm sorry to hear that. 😢 How can I help?"
    elif 'anger' in emotions:
        response += " It's okay to feel angry. Let's talk about it. 😠"
    elif 'gratitude' in emotions:
        response += " You're welcome! 😊"
    
    return response


@app.route('/chat', methods=['POST'])
def chat():
    
    data = request.json
    user_input = data.get('text', '')

    
    emotions = detect_emotion(user_input)

    
    response = generate_response(emotions)

   
    return jsonify({'response': response, 'emotions': emotions})


@app.route('/')
def home():
    return send_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
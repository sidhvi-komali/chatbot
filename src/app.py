from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2
import secrets
import json
import os

app = Flask(__name__)
CORS(app)

# Configure the Gemini API
genai.configure(api_key="AIzaSyArY2nKtfPbJzMc7pBUH_gCaTeR5PdhiAo")
app.secret_key = str(secrets.token_hex(32))  # <-- THIS IS NEEDED!
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

HISTORY_FILE = "chat_history.json"

@app.route("/clear_history", methods=["POST"])
def clear_history():
    with open(HISTORY_FILE, "w") as f:
        f.write("[]")  # empty JSON array
    return jsonify({"message": "History cleared."})

def save_history_to_file(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def load_history_from_file():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

@app.route("/history", methods=["GET"])
def history():
    full_history = load_history_from_file()
    return jsonify(full_history[1:] if len(full_history) > 1 else [])

# Store chat history in session
def get_history():
    if 'history' not in session:
        session['history'] = []
    return session['history']

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    preamble = data.get("preamble", "You are a helpful assistant.")

    # Load history from file (instead of session)
    history = load_history_from_file()
    if not history or history[0].get("preamble") != preamble:
        history = [{"preamble": preamble}]

    history.append({"role": "user", "content": user_input})

    chat = model.start_chat(history=[
        {"role": "user", "parts": [preamble]}
    ] + [
        {"role": h["role"], "parts": [h["content"]]} for h in history[1:]
    ])

    try:
        response = chat.send_message(user_input)
        reply = response.text.strip()
        history.append({"role": "model", "content": reply})
        save_history_to_file(history)
        return jsonify({"reply": reply})
    except Exception as e:
        print("Chat error:", str(e))
        return jsonify({"reply": "Sorry, something went wrong."}), 500


@app.route("/analyze_bruise", methods=["POST"])
def analyze_bruise():
    if 'bruise_image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['bruise_image']
    image = Image.open(file.stream).convert('RGB')  # Convert to RGB for consistency
    image_np = np.array(image)

    # Convert image to HSV for better color segmentation
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Define a range for purple/blue tones (common in bruises)
    lower_bruise = np.array([100, 50, 50])   # HSV lower bound
    upper_bruise = np.array([140, 255, 255]) # HSV upper bound

    # Create a mask for bruise-like areas
    mask = cv2.inRange(hsv, lower_bruise, upper_bruise)

    # Count non-zero pixels in mask (how much of the image might be bruised)
    bruise_pixels = cv2.countNonZero(mask)
    total_pixels = image_np.shape[0] * image_np.shape[1]
    bruise_percentage = (bruise_pixels / total_pixels) * 100

    return jsonify({
        'bruise_percentage': round(bruise_percentage, 2),
        'message': f"{round(bruise_percentage, 2)}% of the image may contain bruising. Your patient must be admitted to the nearest hospital."
    })

if __name__ == "__main__":
    app.run()

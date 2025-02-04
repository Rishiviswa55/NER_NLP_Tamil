from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes and origins
CORS(app)

# Load the trained model, vectorizer, and dictionary from disk
try:
    with open('karky.nerMODEL', 'rb') as model_file:
        classifier = pickle.load(model_file)
    with open('karky.nerVECT', 'rb') as vect_file:
        vectorizer = pickle.load(vect_file)
    with open('karky.nerDICT', 'rb') as dict_file:
        trained_dict = pickle.load(dict_file)
    print("Models and dictionary loaded successfully.")
except Exception as e:
    print(f"Error loading models or dictionary: {e}")
    classifier = None
    vectorizer = None
    trained_dict = {}


def get_ner(word):
    """
    Get the NER label for a given word.

    Parameters:
        word (str): The word to classify.

    Returns:
        label (str): The NER label.
    """
    if not classifier or not vectorizer:
        return "MODEL_NOT_LOADED"

    # Check if the word is already trained
    if word in trained_dict:
        print(f"Word '{word}' found in training dictionary with label '{trained_dict[word]}'.")
        return trained_dict[word]
    else:
        # If the word is not trained, predict the NER
        try:
            word_vector = vectorizer.transform([word])
            prediction = classifier.predict(word_vector)
            predicted_label = prediction[0]
            print(f"Word '{word}' predicted as '{predicted_label}'.")
            return predicted_label
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "PREDICTION_ERROR"


# Define a mapping from NER labels to their corresponding Tamil messages
NER_MESSAGES = {
    'PLACE': (
        "{{{word}}} எனும் சொல் அகராதியில் இல்லை. இது ஒரு இடத்தைக் குறிக்கும் பெயர்ச்சொல்லாக இருக்கலாம் என்று கருதுகிறேன் "
        "எனின் இது ஒரு {{ஊரின் | நாட்டின்}} பெயரைக் குறிக்கலாம்."
    ),
    'MALE': (
        "{{{word}}} எனும் சொல் அகராதியில் இல்லை. இது ஒரு ஆளைக் குறிக்கும் பெயர்ச்சொல்லாக இருக்கலாம் என்று கருதுகிறேன் "
        "எனின் இது ஒரு {{ஆணின்}} பெயரைக் குறிக்கலாம்."
    ),
    'FEMALE': (
        "{{{word}}} எனும் சொல் அகராதியில் இல்லை. இது ஒரு ஆளைக் குறிக்கும் பெயர்ச்சொல்லாக இருக்கலாம் என்று கருதுகிறேன் "
        "எனின் இது ஒரு {{பெண்ணின்}} பெயரைக் குறிக்கலாம்."
    ),
    'PERSON': (
        "{{{word}}} எனும் சொல் அகராதியில் இல்லை. இது ஒரு ஆளைக் குறிக்கும் பெயர்ச்சொல்லாக இருக்கலாம் என்று கருதுகிறேன்."
    ),
    'ORGANIZATION': (
        "{{{word}}} எனும் சொல் அகராதியில் இல்லை. இது ஒரு இடத்தைக் குறிக்கும் பெயர்ச்சொல்லாக இருக்கலாம் என்று கருதுகிறேன் "
        "எனின் இது ஒரு {{ஊரின் | நாட்டின்}} பெயரைக் குறிக்கலாம்."
    ),
    'CURRENCY': (
        "{{{word}}} எனும் சொல் அகராதியில் இல்லை. இது ஒரு நாட்டின் பணத்தைக் குறிக்கும் பெயர்ச்சொல்லாக இருக்கலாம் என்று கருதுகிறேன்."
    ),
    'UNIT': (
        "{{{word}}} எனும் சொல் அகராதியில் இல்லை. இது ஒரு அலகைக் குறிக்கும் பெயர்ச்சொல்லாக இருக்கலாம் என்று கருதுகிறேன் "
        "எனின் இது ஒரு {{எண்ணின் | அளவையின் | தொகையின்}} பெயரைக் குறிக்கலாம்."
    )
}


@app.route('/karefoNER/<word>', methods=['GET'])
def ner_endpoint(word):
    """
    Endpoint to get the NER label for a given word.

    Parameters:
        word (str): The word for which the NER label is to be predicted.

    Returns:
        json: A JSON object containing the input word, its predicted NER label, and an optional message.
    """
    try:
        # Generate the NER label using the model
        label = get_ner(word)

        # Initialize the result dictionary
        result = {'word': word, 'label': label}

        # Check if the word is not in the training dictionary and the label is recognized
        if word not in trained_dict and label in NER_MESSAGES:
            # Format the message with the actual word
            message = NER_MESSAGES[label].format(word=word)
            result['message'] = message

        return jsonify(result), 200

    except Exception as e:
        # Return an error message in case of any exception
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500


# Additional Routes or Functions can be added here

if __name__ == "__main__":
    # Run the Flask app with specified host and port
    app.run(debug=True, host='0.0.0.0', port=3013)

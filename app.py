from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained model
with open("pricing_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict_price():
    # We expect a JSON payload with keys like ACTUAL_AREA, BEDROOMS, PARKING
    # For now, assume numeric features only. If you had categorical features,
    # you need to encode them the same way you did during training.
    
    data = request.get_json()  # Parse the JSON from the request
    
    # Extract features from the request
    # Example: Let's say you only need these three numeric features:
    actual_area = data.get("ACTUAL_AREA", 0)
    bedrooms = data.get("BEDROOMS", 0)
    parking = data.get("PARKING", 0)
    
    # Arrange them into a 2D array (since model.predict expects a list of lists)
    X = [[actual_area, bedrooms, parking]]

    # Make a prediction using the model
    predicted_price = model.predict(X)[0]

    # Return the result as JSON
    return jsonify({"predicted_price": predicted_price})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

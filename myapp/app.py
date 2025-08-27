from flask import Flask, request, jsonify, render_template
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Map numeric prediction → flower name
iris_classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data["features"]  # list of 4 numbers
        prediction = model.predict([features])[0]
        flower_name = iris_classes[int(prediction)]
        return jsonify({"prediction": int(prediction), "flower": flower_name})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

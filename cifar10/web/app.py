import base64
import os

import requests
from flask import Flask, jsonify, render_template, request

MODELS = {
    "baseline": os.getenv("BASELINE_URL", "http://localhost:5051"),
    "optimized": os.getenv("OPTIMIZED_URL", "http://localhost:5052"),
}

app = Flask(__name__)


def predict(url, data_uri):
    resp = requests.post(
        f"{url}/predictions",
        json={"input": {"image": data_uri}},
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    if result.get("status") != "succeeded":
        raise RuntimeError(result.get("error", "Prediction failed."))
    return result["output"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/classify", methods=["POST"])
def classify():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    data = base64.b64encode(file.read()).decode()
    data_uri = f"data:{file.mimetype};base64,{data}"

    results = {}
    for name, url in MODELS.items():
        try:
            results[name] = {"predictions": predict(url, data_uri)}
        except (requests.RequestException, RuntimeError) as e:
            results[name] = {"error": f"Model request failed: {e}"}
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
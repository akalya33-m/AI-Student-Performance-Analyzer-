from flask import Flask, render_template, request
import pickle
import matplotlib
matplotlib.use("Agg")  # ✅ Required for cloud deployment
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "performance_model.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ---------- LOAD MODEL ----------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- ROUTE ----------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    chart_path = None

    if request.method == "POST":
        math = int(request.form["math"])
        reading = int(request.form["reading"])
        writing = int(request.form["writing"])
        attendance = int(request.form["attendance"])

        result = model.predict([[math, reading, writing, attendance]])
        prediction = "PASS 🎉" if result[0] == 1 else "FAIL ❌"

        # Create chart
        scores = [math, reading, writing]
        subjects = ["Math", "Reading", "Writing"]

        plt.figure()
        plt.bar(subjects, scores)
        plt.ylim(0, 100)

        chart_path = os.path.join(STATIC_DIR, "chart.png")
        plt.savefig(chart_path)
        plt.close()

    return render_template("index.html", prediction=prediction, chart="chart.png")

# ---------- RUN SERVER ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Render compatible
    app.run(host="0.0.0.0", port=port, debug=False)

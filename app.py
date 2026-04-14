import os
from flask import Flask, render_template, request
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    image = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            result, confidence = predict_image(path)
            image = path

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           image=image)

if __name__ == "__main__":
    app.run(debug=True)
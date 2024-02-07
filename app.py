from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load CountVectorizer
with open("/workspaces/model-deployment/models/cv.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

# Load Classifier
with open("/workspaces/model-deployment/models/clf.pkl", 'rb') as file:
    clf = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')

    # Check if the loaded vectorizer has a transform method
    if hasattr(vectorizer, 'transform'):
        # Transform the input email using the loaded vectorizer
        tokenized_email = vectorizer.transform([email])

        # Check if the loaded classifier has a predict method
        if hasattr(clf, 'predict'):
            # Predict using the loaded classifier
            prediction = clf.predict(tokenized_email)
            # Assuming clf is a binary classifier, you may need to adjust this logic
            prediction = 1 if prediction[0] == 1 else -1
        else:
            return render_template("index.html", prediction=None, content=email)
    else:
        return render_template("index.html", prediction=None, content=email)

    return render_template("index.html", prediction=prediction, content=email)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

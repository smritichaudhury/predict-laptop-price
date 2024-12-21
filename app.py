from flask import Flask, request, render_template
import pickle
import pandas as pd


app = Flask(__name__)

# Load model, scalers and dummy_columns
try:
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('model/feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")


print("Model and preprocessors loaded successfully.")

def preprocess_input(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_names, fill_value=0)
    scaled_data = scaler.transform(df)
    return scaled_data

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            "ram_expandable": int(request.form.get('ram_expandable', 0)),
            "ram": int(request.form.get('ram', 0)),
            "ghz": float(request.form.get('ghz', 0.0)),
            "display": float(request.form.get('display', 0.0)),
            "ssd": int(request.form.get('ssd', 0)),
            "hdd": int(request.form.get('hdd', 0)),
            "processor": request.form.get('processor', 'Unknown'),
            "ram_type": request.form.get('ram_type', 'Unknown'),
            "display_type": request.form.get('display_type', 'Unknown'),
            "gpu_brand": request.form.get('gpu_brand', 'Unknown')
        }

        preprocessed_input = preprocess_input(input_data)
        prediction = model.predict(preprocessed_input)[0]

        return render_template('index.html', prediction=round(prediction, 2), error=None)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
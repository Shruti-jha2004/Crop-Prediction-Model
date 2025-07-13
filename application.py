from flask import Flask, request, render_template
import numpy as np
import pickle
import os

base_dir = 'D:/Year 3 (2024-25)/AI/Python/jupyter/'  

try:
    model = pickle.load(open(os.path.join(base_dir, 'model.pkl'), 'rb'))
    sc = pickle.load(open(os.path.join(base_dir, 'standscaler.pkl'), 'rb'))
    mx = pickle.load(open(os.path.join(base_dir, 'minmaxscaler.pkl'), 'rb'))
except Exception as e:
    print(f"Error loading model or scalers: {e}")

application = Flask(__name__)

@application.route('/')
def index():
    # index load
    return render_template("index.html")

@application.route("/predict", methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        # Predict the crop
        prediction = model.predict(sc_mx_features)

        # array here
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }

        crop = crop_dict.get(prediction[0], "Sorry, we could not determine the best crop to be cultivated with the provided data.")
        result = f"{crop} is the best crop to be cultivated right there." if crop in crop_dict.values() else crop

        return render_template('index.html', result=result)
    
    except Exception as e:

        error_message = f"An error occurred: {e}"
        print(error_message)
        return render_template('index.html', result=error_message)

if __name__ == "__main__":
    application.run(debug=True)

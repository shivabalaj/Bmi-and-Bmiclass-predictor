from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained models
bmi_model = joblib.load('hight and weight predicter/bmi_model.joblib')
bmi_class_model = joblib.load('hight and weight predicter/bmi_class_model.joblib')
le = joblib.load('hight and weight predicter/label_encoder.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])

        # Prepare the input data for the models
        input_data = np.array([[age, height, weight]])

        # Predict BMI and BMI Class
        bmi_pred = bmi_model.predict(input_data)[0]
        bmi_class_pred = bmi_class_model.predict(input_data)[0]
        bmi_class_name = le.inverse_transform([bmi_class_pred])[0]

        return render_template('index.html', 
                               bmi=bmi_pred, 
                               bmi_class=bmi_class_name, 
                               age=age, 
                               height=height, 
                               weight=weight)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load('loan_model.pkl')
@app.route('/')
def home():
    return render_template('form.html', prediction=None)
@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    try:
        values = [
            float(request.form['Gender']),
            float(request.form['Married']),
            float(request.form['Dependents']),
            float(request.form['Education']),
            float(request.form['Self_Employed']),
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            float(request.form['Property_Area']),
        ]
        prediction = model.predict([values])[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"
        return render_template('form.html', prediction=result)
    except Exception as e:
        return f"Error: {e}"
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    features = np.array([list(data.values())])
    prediction = model.predict(features)
    return jsonify({'Loan_Status': str(int(prediction[0]))})
if __name__ == '__main__':
    app.run(debug=True)

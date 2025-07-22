import requests
# Flask API URL
url = "http://127.0.0.1:5000/predict"
# Sample data (must match model input format)
data = {
    "Gender": 1,
    "Married": 1,
    "Dependents": 0,
    "Education": 0,
    "Self_Employed": 0,
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 128.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": 2
}
# Send POST request to API
response = requests.post(url, json=data)
# Show the response
print("Prediction Result:", response.json())

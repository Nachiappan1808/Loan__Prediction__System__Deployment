from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('loan-prediction-lr-model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        Gender = float(request.form['Gender'])
        Married = float(request.form['Married'])
        Dependents = float(request.form['Dependents'])
        Education = int(request.form['Education'])
        Self_Employed = float(request.form['Self_Employed'])
        appincome = int(request.form['appincome'])
        coappincome = float(request.form['coappincome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Property_Area = int(request.form['Property_Area'])
        Credit_History = float(request.form['Credit_History'])
        data = np.array([[Gender,Married,Dependents,Education,Self_Employed,appincome,coappincome,LoanAmount,Loan_Amount_Term,Property_Area,Credit_History]])
        my_prediction = model.predict(data)
        
        if(my_prediction==1):
            val="Yes it will!!"
        else:
            val="No.Please try another bank!!"
    

    return render_template('index.html', prediction_text=val)


if __name__ == "__main__":
    app.run(debug=True)
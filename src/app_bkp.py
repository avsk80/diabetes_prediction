# importing the necessary dependencies
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])  # route to display the home page
@cross_origin()
def predicting():
    try:
        pregnancies = float(request.form['pregnancies'])
        diabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
        insulin = float(request.form['insulin'])
        glucose = float(request.form['Glucose'])
        bp = float(request.form['BP'])
        bmi = float(request.form['BMI'])
        skin_thickness = float(request.form['skin_thickness'])
        file_name_model = "diabetes_predict.sav"
        load_model = pickle.load(open(file_name_model, 'rb'))
        scaler = pickle.load(open("scaler_diabetes.sav", 'rb'))
        prediction = load_model.predict(scaler.fit_transform([pregnancies,diabetesPedigreeFunction, age,insulin,glucose,bp,bmi,skin_thickness]))
        print('prediction is', prediction)
        print('prediction is', prediction[0])
        print('prediction is', prediction[0][0])
        # showing the prediction results in a UI
        return render_template('results.html', prediction=round(10 * prediction[0][0]))
    except Exception as e:
        print('The Exception message is: ', e)
        return 'something is wrong'
@app.route('/predict_api', methods=['POST'])  # route to display the home page
@cross_origin()
def predicting_api():
    try:
        gre_score = float(request.json['gre_score'])
        toefl_score = float(request.json['toefl_score'])
        university_rating = float(request.json['university_rating'])
        sop = float(request.json['sop'])
        lor = float(request.json['lor'])
        cgpa = float(request.json['cgpa'])
        is_research = request.json['research']
        if is_research == "yes":
            research = 1
        else:
            research = 0
        file_name_model = "admission_model.pickle"
        load_model = pickle.load(open(file_name_model, 'rb'))
        #scaler = pickle.load(open("scaler_model.pickle", 'rb'))
        prediction = load_model.predict([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])
        print('prediction is', prediction)
        print('prediction is', prediction[0])
        print('prediction is', prediction[0][0])
        return jsonify({"prediction": 10 * prediction[0][0]})
    except Exception as e:
        print('The Exception message is: ', e)
        return 'something is wrong'


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)
    #app.run(debug=True)


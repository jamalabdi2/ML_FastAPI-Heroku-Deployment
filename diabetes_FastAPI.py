from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json



class model_input(BaseModel):
    Pregnancies: int                 
    Glucose: int                       
    BloodPressure: int                
    SkinThickness: int                 
    Insulin: int                      
    BMI: float                         
    DiabetesPedigreeFunction: float    
    Age: float   


app = FastAPI()
#loading the saved model  

@app.post('/diabetes_prediction')
 
def predict(input_parameters: model_input):
    preg = input_parameters.Pregnancies
    glu = input_parameters.Glucose
    bp = input_parameters.BloodPressure
    skin = input_parameters.SkinThickness
    ins = input_parameters.Insulin
    bmi = input_parameters.BMI
    dpf = input_parameters.DiabetesPedigreeFunction
    age = input_parameters.Age

    feature = list([preg,glu,bp,skin,ins,bmi,dpf,age])
    model = pickle.load(open('diabetes_randomforest_model.pkl','rb'))
    prediction = model.predict([feature])
    probability = model.predict_proba([feature]) 

    if (prediction[0]==1):
        return{'Prediction: ':f'You have been tested positive with {probability[0][1]} probability'}
    else:
        return{'Prediction: ':f'You have been tested negative with {probability[0][0]} probability'}



 

     
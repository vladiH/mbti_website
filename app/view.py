from app import app

from flask import Flask, render_template, url_for, request, redirect
from app.models.load_model import LoadModel
from app.clasess.cmbti import MBTI

def load():
    model = LoadModel()
    model.load()
    return model

model = load()
cmbti = MBTI()

@app.route('/', methods=['POST','GET'])
def index():
    if request.method =='POST':
        user_query = request.form['subject']
        label_mbti, percent = model.predict(user_query)
        print(label_mbti)
        cmbti.labels = label_mbti
        cmbti.values = percent
        return render_template('publics/result.html', title=cmbti.name, max=17000, set=zip(cmbti.values,cmbti.labels,cmbti.colores))
    else:
        return render_template('publics/index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('publics/about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('publics/contact.html')
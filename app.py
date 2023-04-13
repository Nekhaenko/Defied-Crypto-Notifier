from flask import Flask, render_template, request, jsonify
from markupsafe import escape

import pandas as pd
import joblib
import json

with open('config.json') as json_data:
    config = json.load(json_data)

text_model = config['text_model']
lr = config['lr_model']

app = Flask(__name__)

text_model = joblib.load(text_model)
lr = joblib.load(lr)


@app.route('/', methods=['post', 'get'])
def login():
    message = ''
    if request.method == 'POST':
        text = request.form.get('text')

        if len(text) > 0:
            df=pd.DataFrame({'text':[text]})
            text = text_model.transform(df.text)
            message = str(lr.predict(text)[0])
            print(message)
        else:
            message = "Wrong text"

    return render_template('index.html', message=message)


if __name__=='__main__':
    app.run()
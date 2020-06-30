from flask import Flask, render_template, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms import validators

import pandas as pd
import joblib

import string
from nltk.corpus import stopwords
#from tensorflow.keras.models import load_model




def return_emotion(ppln,emo_examp):
    
    df = pd.DataFrame(emo_examp.items(), columns = ['name', 'txt'])
    
    class_ind = ppln.predict(df['txt'])
    
    if class_ind[0] == 'joy':
        return 'Joy!'
    elif class_ind[0] == 'sadness':
        return 'Sadness!'
    elif class_ind[0] == 'anger':
        return 'Anger!'
    elif class_ind[0] == 'fear':
        return 'Fear!'
    elif class_ind[0] == 'love':
        return 'Love!'
    else:
        return 'Surprise!'



def text_process(mess):

    nopunc = [char for char in mess if char not in string.punctuation]
    
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]



app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'mysecretkey'

# LOAD THE MODEL!
loaded_pipeline = joblib.load('emo_pipeline.pkl')

# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
class EmotionForm(FlaskForm):
    Text = TextAreaField('Text', [validators.required(), validators.length(min=5,max=1000)], 
                                  render_kw={"placeholder": "Enter your text here..", "size":300})
    submit = SubmitField('Predict')

@app.route('/', methods=['GET', 'POST'])
def index():

    # Create instance of the form.
    form = EmotionForm()
    # If the form is valid on submission (we'll talk about validation next)
    if form.validate_on_submit():
        # Grab the data from the breed on the form.

        session['text'] = form.Text.data

        return redirect(url_for("prediction"))


    return render_template('home.html', form = form)


@app.route('/prediction')
def prediction():

    content = {}

    content['text'] = session['text']

    results = return_emotion(ppln = loaded_pipeline, emo_examp = content)
    
    if results == 'Joy!':
        return render_template('prediction_joy.html',results_joy=results)
    elif results == 'Sadness!':
        return render_template('prediction_sadness.html',results_sadness=results)
    elif results == 'Anger!':
        return render_template('prediction_anger.html',results_anger=results)
    elif results == 'Fear!':
        return render_template('prediction_fear.html',results_fear=results)
    elif results == 'Love!':
        return render_template('prediction_love.html',results_love=results)
    else:
        return render_template('prediction_surprise.html',results_surprise=results)

if __name__ == '__main__':
    app.run(debug=True)

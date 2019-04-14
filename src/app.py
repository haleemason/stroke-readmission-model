import os
import pydf
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from goog_utils import load_data
from tf_utils import initialize_parameters, forward_propagation, scale_feature
import lime
import lime.lime_tabular
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, validators


print("reached -1")

# App config
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'b6c940edbcecb2d2c37823f912c6d7a94904bc9e0664ca40'


class ReusableForm(Form):
    pay1_private = TextField('PrivateInsurance:', validators=[validators.required()])
    metro = TextField('Metro:', validators=[validators.required(), validators.Length(min=1, max=3)])
    diabetes = TextField('Diabetes:', validators=[validators.required(), validators.Length(min=1, max=3)])
    copd = TextField('COPD:', validators=[validators.required(), validators.Length(min=1, max=3)])
    ckd = TextField('CKD:', validators=[validators.required(), validators.Length(min=1, max=3)])
    chf = TextField('CHF:', validators=[validators.required(), validators.Length(min=1, max=3)])
    atrial_fib = TextField('AFib:', validators=[validators.required(), validators.Length(min=1, max=3)])
    hyperlipidemia = TextField('Hyperlipidemia:', validators=[validators.required(), validators.Length(min=1, max=3)])
    sex = TextField('Sex:', validators=[validators.required(), validators.Length(min=1, max=6)])
    nicotine = TextField('Nicotine:', validators=[validators.required(), validators.Length(min=1, max=3)])
    obesity = TextField('Obesity:', validators=[validators.required(), validators.Length(min=1, max=3)])
    hypertension = TextField('Hypertension:', validators=[validators.required(), validators.Length(min=1, max=3)])
    age = TextField('Age:', validators=[validators.required(), validators.Length(min=1, max=3)])


@app.route("/", methods=['GET', 'POST'])
@app.route("/explain")
def explain():
    form = ReusableForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        pay1_private = request.form['pay1_private']
        metro = request.form['metro']
        diabetes = request.form['diabetes']
        copd = request.form['copd']
        ckd = request.form['ckd']
        chf = request.form['chf']
        atrial_fib = request.form['atrial_fib']
        hyperlipidemia = request.form['hyperlipidemia']
        sex = request.form['sex']
        nicotine = request.form['nicotine']
        obesity = request.form['obesity']
        hypertension = request.form['hypertension']
        age = request.form['age']
        
        pay1_private = int(pay1_private[0].lower() == 'y')
        metro = int(metro[0].lower() == 'y')
        diabetes = int(diabetes[0].lower() == 'y')
        copd = int(copd[0].lower() == 'y')
        ckd = int(ckd[0].lower() == 'y')
        chf = int(chf[0].lower() == 'y')
        atrial_fib = int(atrial_fib[0].lower() == 'y')
        hyperlipidemia = int(hyperlipidemia[0].lower() == 'y')
        sex = int(sex[0].lower() == 'f')
        nicotine = int(nicotine[0].lower() == 'y')
        obesity = int(obesity[0].lower() == 'y')
        hypertension = int(hypertension[0].lower() == 'y')
        age = int(age)
        
        patient = [pay1_private, metro, diabetes, copd, ckd, chf, atrial_fib, age, hyperlipidemia, sex, nicotine, obesity, hypertension]

# *********************************** #
        
        features = ['readm90day', 'pay1_private', 'metro', 'diabetes', 'copd', 'ckd', 'chf', 'atrial_fib', 
                    'age', 'hyperlipidemia', 'sex', 'nicotine','obesity', 'hypertension']
        filename = "./final_stroke_dx1.csv"

        # Load the train, test, and validation data
        (X_scaled_train, X_scaled_test,
         X_scaled_valid, Y_scaled_train,
         Y_scaled_test, Y_scaled_valid, X_scaler) = load_data(filename, features)
        
        layer_1_nodes = 37
        
        number_of_inputs = X_scaled_train.shape[1]
        number_of_outputs = int(len(np.unique(Y_scaled_train)))
        
        ops.reset_default_graph()
        
        # Input
        with tf.variable_scope('input'):
            X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
        
        with tf.variable_scope('output'):
            parameters = initialize_parameters(number_of_inputs, layer_1_nodes, number_of_outputs)
            logits = forward_propagation(X, parameters)
                
        with tf.variable_scope('save'):
            saver = tf.train.Saver()
            
        # Initialize a session so that we can run TensorFlow operations
        with tf.Session() as session:
        
            # Load trained model from disk:
            saver.restore(session, "./exportedmodel/trained_variables.ckpt")
        
            print("Trained model loaded from disk.")
            
            categorical_columns = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
            feature_names_categorical = ['pay1_private', 'metro', 'diabetes', 'copd', 'ckd', 'chf', 'atrial_fib', 
                                         'hyperlipidemia', 'sex', 'nicotine','obesity', 'hypertension']
                
            feature_names_all = ['pay1_private', 'metro', 'diabetes', 'copd', 'ckd', 'chf', 'atrial_fib', 
                                 'age', 'hyperlipidemia', 'sex', 'nicotine','obesity', 'hypertension']
            
            # Create the LIME Explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled_train,
                                                               feature_names=feature_names_all,
                                                               class_names=[0, 1],
                                                               categorical_features=categorical_columns,
                                                               categorical_names=feature_names_categorical,
                                                               kernel_width=3)
        
            def predict_func(X_valid):
                """Return probabilities for a binary class
                For binary classification only."""
                
                # Convert logits to probabilities
                softmax_probas = tf.nn.softmax(logits)
                actual_probas = softmax_probas.eval(feed_dict={X: X_valid})
                
                return actual_probas

            # Create a function that return prediction probabilities
            predictor = lambda x: predict_func(x).astype(float)
            
            patient = scale_feature(patient, X_scaler)
            
            exp = explainer.explain_instance(patient, predictor, num_features=14)
            cdir = os.path.dirname(os.path.abspath(__file__))
            currfile = cdir + '/static/img/exp.pdf'
            try:
                os.remove(cdir + '/templates/lime.html')
                os.remove(currfile)
            except OSError:
                pass
        
            exp.save_to_file("./templates/lime.html")
            
            print("reached 0")
            
            #imgkit.from_url("./templates/lime.html", './static/img/exp.jpg')
            #pdfkit.from_file("./templates/lime.html", './static/img/exp.pdf')

            pdf = pydf.generate_pdf(open('./templates/lime.html', encoding='utf-8').read())
            with open(currfile, 'wb') as f:
                f.write(pdf)


# *********************************** #
 
        if form.validate():
            #webbrowser.open("./templates/lime.html", new=2)
            return render_template('index.html', form=form)
        
        else:
            flash('Error: All the form fields are required. ')
 
    return render_template('explain.html', form=form)


if __name__ == "__main__":
    app.run()

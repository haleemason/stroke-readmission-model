import numpy as np
import pandas as pd
import tensorflow as tf
#import googleapiclient.discovery
from sklearn import preprocessing
#from oauth2client.client import GoogleCredentials
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular


seed = 0

def predict_proba(inputs_array, logits):
    """Given logits (i.e. unnormalized log probabilities), output the normalized 
       linear probabilities of the classes. (Note: Classes are ordered as they are 
       in active_features_ in sklearn.preprocessing.OneHotEncoder).
       For binary classification only.
	
     Arguments:
     logits -- A dictionary of tensorflow logits return by Google ML Cloud Service. 
     E.g. [{'readm': [0.0969928503036499, 0.009187504649162292]}]

     Return:
     An array of probabilities of shape (1, 2) for the two classes: readm90day yes/no
    """
    # Convert logits to probabilities
    lgts_list = []
    X = tf.placeholder(tf.float32, shape=(None, 13))
    with tf.Session() as session:
        for lgts in logits:
            lgts = lgts['readm']
            lgts_list.append(lgts)
        softmax_probas = tf.nn.softmax(lgts_list)
        if len(inputs_array.shape) == 1:
            inputs_array = inputs_array.reshape(1, inputs_array.shape[0])
        actual_probas = session.run(softmax_probas, feed_dict={X: inputs_array})
    return actual_probas
    

def format_2d_input(inputs):
    """Formats numpy array as serializable JSON object suitable for 
        requesting online predictions from gcloud ml-engine projects. E.g.
        instances = [{"input": [1,2, 3]}, {"input": [1,2, 3]}]. 
        Reference: https://goo.gl/6wZLPE
        
        Arguments:
        numpy array
        
        Return:
        JSON object
        """
    formatted_inputs = []
    for each in inputs:
        formatted_inputs.append({"input": each.tolist()})
    return formatted_inputs

def predict_fn_neuralnet(inputs_array):
    """ An interface that make API call to gcloud ml-engine with a numpy array
        containing instances for which we want to return predictions.
        
        Argument:
        numpy array of shape (sample size, 13) corresponding to number of patients and the features.
        
        Return:
        the probabilities for the input data
    """
    if len(inputs_array.shape) == 1:
        inputs_for_prediction = {"input": inputs_array.tolist()}
    else:
        inputs_for_prediction = format_2d_input(inputs_array)

    PROJECT_ID = "readmission-179223"
    MODEL_NAME = "readm"
    CREDENTIALS_FILE = "credentials.json"
    
    # Connect to the Google Cloud-ML Service
    credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
    service = googleapiclient.discovery.build('ml', 'v1', credentials=credentials)
    
    # Connect to our Prediction Model
    name = 'projects/{}/models/{}'.format(PROJECT_ID, MODEL_NAME)
    response = service.projects().predict(
        name = name,
        body={'instances': inputs_for_prediction}
    ).execute()
    
    # Report any errors
    if 'error' in response:
        raise RuntimeError(response['error'])
    
    # Grab the results from the response object
    logits_dict = response['predictions'] 
    
    return predict_proba(inputs_array, logits_dict)


def one_hot_matrix(target_vector):
    """
    Creates a one hot matrix. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix where [1., 0.] represents 0 label and [0., 1.]
    represents 1 label
    """
    enc = preprocessing.OneHotEncoder()
    enc.fit(target_vector)
    onehotlabels = enc.transform(target_vector).toarray()
    return onehotlabels

def oversample(X_train, Y_train, seed=None):
    """oversample the positive class"""
    alldata = np.c_[X_train, Y_train]
    df = pd.DataFrame(alldata)
    positives = df[df.iloc[:, -1] == 1]
    negatives = df[df.iloc[:, -1] == 0]
    positives = positives.sample(n=len(negatives), replace=True, random_state=seed)
    newdata = pd.concat([positives, negatives])
    Y_train = newdata.iloc[:, -1].values
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    X_train = newdata.iloc[:, :-1].values
    return X_train, Y_train
    
def load_data(filename, colnames):
    """preprocess data for a neural network"""
    data = pd.read_csv(filename, usecols=colnames)
    data = data[colnames]

    # Drop the few missing values
    data = data.dropna(axis=0, how='any')
    
    # Preprocess data
    data = data.sample(frac=1, random_state=seed)
    array = data.values
    X_raw = array[:, 1:]
    Y_raw = array[:, 0]
    Y_raw = Y_raw.astype(int)
    Y_raw = Y_raw.reshape(Y_raw.shape[0], 1)
        
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_raw, Y_raw, test_size=0.05, random_state=seed)
    X_test, X_valid, Y_test, Y_valid = train_test_split(X_test, Y_test, test_size=0.5, random_state=seed)
    
    # Resample the data
    X_train, Y_train = oversample(X_train, Y_train, seed)
    
    # All data needs to be scaled to a small range like 0 to 1 for the neural
    # network to work well. Create scalers for the inputs and outputs.
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale both the training inputs and outputs
    X_scaled_train = X_scaler.fit_transform(X_train)
    Y_scaled_train = Y_scaler.fit_transform(Y_train)
    
    # It's very important that the training and test data are scaled with the same scaler.
    X_scaled_test = X_scaler.transform(X_test)
    Y_scaled_test = Y_scaler.transform(Y_test)
    
    # Likewise for the validation set
    X_scaled_valid = X_scaler.transform(X_valid)
    Y_scaled_valid = Y_scaler.transform(Y_valid)
    
    # One hot encoding of the output
    Y_scaled_train = one_hot_matrix(Y_scaled_train)
    Y_scaled_train = np.squeeze(Y_scaled_train)
    Y_scaled_test = one_hot_matrix(Y_scaled_test)
    Y_scaled_test = np.squeeze(Y_scaled_test)
    Y_scaled_valid = one_hot_matrix(Y_scaled_valid)
    Y_scaled_valid = np.squeeze(Y_scaled_valid)

    return X_scaled_train, X_scaled_test, X_scaled_valid, Y_scaled_train, Y_scaled_test, Y_scaled_valid, X_scaler
    
def explain_prediction():
    """Loads and processes the data for lime"""
    
    features = ['readm90day', 'pay1_private', 'metro', 'diabetes', 'copd', 'ckd', 'chf', 'atrial_fib', 
                'age', 'hyperlipidemia', 'sex', 'nicotine','obesity', 'hypertension']
    filename = "final_stroke_dx1.csv"
    (   X_scaled_train, X_scaled_test, 
         X_scaled_valid, Y_scaled_train, 
         Y_scaled_test, Y_scaled_valid, X_scaler   
    ) = load_data(filename, features)
    
    categorical_columns = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    feature_names_categorical = ['pay1_private', 'metro', 'diabetes', 'copd', 'ckd', 'chf', 'atrial_fib', 
                                 'hyperlipidemia', 'sex', 'nicotine','obesity', 'hypertension']

    feature_names_all = ['pay1_private', 'metro', 'diabetes', 'copd', 'ckd', 'chf', 'atrial_fib', 
                         'age', 'hyperlipidemia', 'sex', 'nicotine','obesity', 'hypertension']
    
    #Create a function that return prediction probabilities
    predictor = lambda x: predict_fn_neuralnet(x).astype(float)
    
    #Create the LIME Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X_scaled_train ,feature_names = feature_names_all, class_names=[0,1],
                                                   categorical_features=categorical_columns, 
                                                   categorical_names=feature_names_categorical, kernel_width=3)
    return explainer, predictor, X_scaler

    
def scale_feature(feature, scaler):
    """Use a fitted scaler to rescale features"""
    feature = scaler.transform([feature])
    return feature.ravel()
    

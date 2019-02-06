import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize


seed = 0


def random_batches(X, Y, batch_size = 64, seed = seed):
    """
    Creates a list of random batches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector, of shape (n_classes, number of examples)
    batch_size - size of the batches, integer
    seed -- this is only for the purpose of grading, so that you're "random batches are the same as ours.
    
    Returns:
    batches -- list of synchronous (batch_X, batch_Y)
    """
    
    m = X.shape[0]
    batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_batches = int(np.floor(m / batch_size))
    for k in range(0, num_complete_batches):
        batch_X = shuffled_X[k * batch_size : k * batch_size + batch_size, :]
        batch_Y = shuffled_Y[k * batch_size : k * batch_size + batch_size, :]
        mini_batch = (batch_X, batch_Y)
        batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        batch_X = shuffled_X[num_complete_batches * batch_size : m, :]
        batch_Y = shuffled_Y[num_complete_batches * batch_size : m, :]
        mini_batch = (batch_X, batch_Y)
        batches.append(mini_batch)
    
    return batches



def initialize_parameters(number_of_inputs, layer_1_nodes, number_of_outputs):
    # Input layer
    W1 = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b1 = tf.get_variable(name="biases1", shape=[1, layer_1_nodes], initializer=tf.zeros_initializer())
    
    # hidden layer 1
    W2 = tf.get_variable("weights2", shape=[layer_1_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b2 = tf.get_variable(name="biases2", shape=[1, number_of_outputs], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    
def initialize_parameters_3nn(number_of_inputs, layer_1_nodes, layer_2_nodes, layer_3_nodes, number_of_outputs):
    # Input layer
    with tf.variable_scope('layer_1'):
        W1 = tf.get_variable("weights1", shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b1 = tf.get_variable(name="biases1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    
    # Hidden Layer 1
    with tf.variable_scope('layer_2'):
        W2 = tf.get_variable("weights2", shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b2 = tf.get_variable(name="biases2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    
    # Hidden layer 2
    with tf.variable_scope('layer_3'):
        W3 = tf.get_variable("weights3", shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b3 = tf.get_variable(name="biases3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    
    # Hidden Layer 3
    with tf.variable_scope('output'):
        W4 = tf.get_variable("weights4", shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
        b4 = tf.get_variable(name="biases4", shape=[1, number_of_outputs], initializer=tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}
    
    return parameters

def forward_propagation_3nn(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    # layer 1
    A1 = tf.nn.relu(tf.matmul(X, W1) + b1)   
        
    # layer 2
    A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)

    # layer 3
    A3 = tf.nn.relu(tf.matmul(A2, W3) + b3)

    # layer 3
    A4 = tf.nn.relu(tf.matmul(A3, W4) + b4)
    
    prediction = A4
    
    return prediction


def forward_propagation(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # layer 1
    A1 = tf.nn.relu(tf.matmul(X, W1) + b1)    
        
    # layer 2
    A2 = tf.nn.relu(tf.matmul(A1, W2) + b2)
    
    return A2
    
def compute_cost(logits, labels):
    """
    Computes the cost
    
    Arguments:
    fa -- output of forward propagation (activation of the last LINEAR unit), of shape (n_features, m)
    y -- "true" labels vector placeholder, same shape as fa
    
    Returns:
    cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    #logits = tf.transpose(logits)
    #labels = tf.transpose(labels)
    
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) )

def compute_cost_with_reg(logits, labels, parameters, beta = 0.01):

    # Retrieve the weights from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) + 
                          beta * tf.nn.l2_loss(W1) + beta * tf.nn.l2_loss(W2))

def compute_cost_weighted(Y, logits, class_weight):
    
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=Y, logits=logits, pos_weight=class_weight))


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

def class_balanced_cross_entropy(pred, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    Args:
        pred: of shape (b, ...). the predictions in [0,1].
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_cross_entropy'):
        z = batch_flatten(pred)
        y = tf.cast(batch_flatten(label), tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        eps = 1e-12
        loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
        loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
    cost = tf.subtract(loss_pos, loss_neg, name=name)
    return cost

def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)


def model(X_train, X_test, Y_train, Y_test, layer_1_nodes, learning_rate = 0.0001, 
                         num_epochs = 10000, batch_size = 32, print_cost = True, save=False):
    seed = 0
    m = X_train.shape[0]
    number_of_outputs = Y_train.shape[1]
    number_of_inputs = X_train.shape[1]
    
    # Create Placeholders
    X = tf.placeholder(tf.float32, [None, number_of_inputs])
    Y = tf.placeholder(tf.float32, [None, number_of_outputs])
    
    # Initialize parameters
    parameters = initialize_parameters(number_of_inputs, layer_1_nodes, number_of_outputs)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z2 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z2, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    costs = []
    costs_df = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / batch_size)
            seed += 1
            minibatches = random_batches(X_train, Y_train, batch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (batch_x, batch_y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if epoch % 10 == 0:
                costs.append(epoch_cost)
            if epoch % 100 == 0:
                costs_df.append(epoch_cost)
            if print_cost and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # metrics
        correct_prediction = tf.equal(tf.argmax(Z2, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        y_pred = tf.argmax(Z2, 1)
        
        # train
        accuracy_train, y_pred_train = sess.run([accuracy, y_pred], feed_dict={X: X_train, Y: Y_train})
        y_true_train = np.argmax(Y_train, 1)
        f1_score_train = metrics.f1_score(y_true_train, y_pred_train)
        
        # test
        accuracy_test, y_pred_test = sess.run([accuracy, y_pred], feed_dict={X: X_test, Y: Y_test})
        y_true_test = np.argmax(Y_test, 1)
        f1_score_test = metrics.f1_score(y_true_test, y_pred_test)
                
        # output metrics
        if print_cost == True:
            print ("Train Accuracy:", accuracy_train)
            print ("Test Accuracy:", accuracy_test)
            print ("Train f1 score:", f1_score_train)
            print ("Test f1 score:", f1_score_test)
 
        eval_metrics = {"accuracy_train": accuracy_train, "accuracy_test": accuracy_test,
                   "f1_train": f1_score_train, "f1_test": f1_score_test}
        
        saver = tf.train.Saver()
        if save:
            save_path = saver.save(sess, "logs/model.ckpt")
            print ("Model saved in file:", save_path)
        
        return eval_metrics, costs_df, parameters

def eval_metrics_old_version(logits, X_train, Y_train, X_test, Y_test, X, session):
    """ouputs evaluation metrics from the logits and data"""
    Y_pred_test = session.run(logits, feed_dict={X: X_test})
    Y_pred_test = np.argmax(Y_pred_test, 1)
    Y_pred_train = session.run(logits, feed_dict={X: X_train})
    Y_pred_train = np.argmax(Y_pred_train, 1)

    # Metrics: test
    Y_true_test = np.argmax(Y_test, 1)
    accuracy_test = metrics.accuracy_score(Y_true_test, Y_pred_test)
    f1_score_test = metrics.f1_score(Y_true_test, Y_pred_test)

    # Metrics: train
    Y_true_train = np.argmax(Y_train, 1)
    f1_score_train = metrics.f1_score(Y_true_train, Y_pred_train)
    accuracy_train = metrics.accuracy_score(Y_true_train, Y_pred_train)

    return accuracy_train, accuracy_test, f1_score_train, f1_score_test

def eval_metrics(logits, X_train, Y_train, X_test, Y_test, X, session, threshold=0.5):

    #number_of_inputs = X_train.shape[1]
    #X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))
    
    # Convert logits to linear probabilities
    softmax_probas = tf.nn.softmax(logits)
    actual_probas_train = session.run(softmax_probas, feed_dict={X: X_train})
    actual_probas_test = session.run(softmax_probas, feed_dict={X: X_test})
                
    # Compute evaluation metrics on the train set
    y_true_train = np.argmax(Y_train, 1)
    y_true_train = y_true_train.reshape(y_true_train.shape[0], 1)
    y_pred_prob_train = actual_probas_train[:, 1] # store the predicted probabilities for class 1
    y_pred_train = binarize(y_pred_prob_train, threshold)[0]
    y_pred_train = y_pred_train.reshape(y_pred_train.shape[0], 1)
    accuracy_train = metrics.accuracy_score(y_true_train, y_pred_train)
    recall_train = metrics.recall_score(y_true_train, y_pred_train)
    precision_train = metrics.precision_score(y_true_train, y_pred_train)
    f1_score_train = metrics.f1_score(y_true_train, y_pred_train)
    
    # Compute evaluation metrics on the test set
    y_true_test = np.argmax(Y_test, 1)
    y_true_test = y_true_test.reshape(y_true_test.shape[0], 1)
    y_pred_prob_test = actual_probas_test[:, 1] # store the predicted probabilities for class 1
    y_pred_test = binarize(y_pred_prob_test, threshold)[0]
    y_pred_test = y_pred_test.reshape(y_pred_test.shape[0], 1)
    accuracy_test = metrics.accuracy_score(y_true_test, y_pred_test)  
    recall_test = metrics.recall_score(y_true_test, y_pred_test)
    precision_test = metrics.precision_score(y_true_test, y_pred_test)
    f1_score_test = metrics.f1_score(y_true_test, y_pred_test)

    metrics_df = pd.DataFrame(columns=["accuracy", "recall", "precision", "f1_score"])
    metrics_df.loc[0] = [accuracy_train, recall_train, precision_train, f1_score_train]
    metrics_df.loc[1] = [accuracy_test, recall_test, precision_test, f1_score_test]
    metrics_df.index = ["train", "dev"]
    
    return (metrics_df, accuracy_train, accuracy_test, 
            recall_train, recall_test, 
            precision_train, precision_test,
            f1_score_train, f1_score_test)
    
    
def predict(logits, X_valid, Y_valid, X):
    
    '''return probabilities for a binary class
    For binary classification only.'''
    
    # Convert logits to probabilities
    softmax_probas = tf.nn.softmax(logits)
    actual_probas = softmax_probas.eval(feed_dict={X: X_valid})

    #Y_pred_valid_tf = session.run(logits, feed_dict={X: X_valid})
    Y_pred_valid = np.argmax(actual_probas, 1)
    
    # Metrics: Validation set
    Y_true_valid = np.argmax(Y_valid, 1)
    
    # output true labels, predicted labels, and probabilities  
    probas_list = []
    for i, _ in enumerate(actual_probas):
        if Y_pred_valid[i]: 
            proba = [ Y_pred_valid[i], Y_true_valid[i], actual_probas[i][1]  ]
        else:
            proba = [ Y_pred_valid[i], Y_true_valid[i], actual_probas[i][0]  ]
        probas_list.append(proba)
    probas_df = pd.DataFrame(probas_list, columns=['predicted', 'actual', 'probability'])
    
    return probas_df, Y_true_valid, Y_pred_valid
    

def scale_feature(feature, scaler):
    """Use a fitted scaler to rescale features"""
    feature = scaler.transform([feature])
    return feature.ravel()

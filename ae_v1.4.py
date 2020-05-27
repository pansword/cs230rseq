#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version 1.1: AE with Xavier initializer, no bias and MSE as loss
version 1.2: Changed MSE to ZINB loss function
version 1.3: Changed to Keras
version 1.4: Changed bottleneck to relu, added dropout regularization, l2 regularizer


"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from loss import ZINB, NB
from keras import backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow.keras import regularizers

#expr = pd.read_csv('data/Bladder-counts.1.csv')

#expr = pd.read_csv('data/Bladder-counts.1.csv')
expr = pd.read_csv('data/five_tissue_annotate_color.txt', sep='\t')
#expr = expr.iloc[0:100,:]

 
#csv_dataset = pd.read_csv("data/Bladder-counts.csv", header=0).transpose()
#Remove the first row of the array
#dataset_y = csv_dataset.index[1:]

#Get the features into X
#dataset_x = csv_dataset.iloc[1:,:]

X = expr.values[:,1:(expr.shape[1]-1)].astype(np.float64)
cell_ontology_class = expr.values[:,expr.shape[1]-1]
X = np.log(X + 1) 
#epsilon=np.random.rand(X.shape[0],X.shape[1])*(10**-8)
#X = X + epsilon
expr['Cluster'] = expr['tissue'].apply(lambda x:1 if x=='Heart' else 2 if x=='Bladder' else 3 if x=='Brain_Myeloid' else 4 if x=='Brain_Non-Myeloid' else 5)
Y = expr.values[:,expr.shape[1]-1]


#START


#Performing PCA and projection on tsne
n_input = 30
x_train = PCA(n_components = n_input).fit_transform(X)
y_train=Y
#tsne on PCA
model_tsne = TSNE(learning_rate = 100, n_components = 2, random_state = 123,
                  perplexity = 90, n_iter = 1000, verbose = 1)
tsne = model_tsne.fit_transform(x_train)
expr['tsne_1'] = tsne[:,0]
expr['tsne_2'] = tsne[:,1]
plt.figure(figsize=(10,8))
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="tissue",
    palette=sns.color_palette("bright", 5),
    data=expr,
    legend="full",
    alpha=0.3
)

ax = plt.gca()
ax.set_title("tsne on Keras implementation")

#END


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#unique_types = list(set(cell_ontology_class))
#numerical_labels = np.zeros((len(cell_ontology_class),1)) 
#numerical_labels[cell_ontology_class==unique_types[0]] = 0.5
#numerical_labels[cell_ontology_class==unique_types[1]] = 1
#Y = numerical_labels
#Y = Y.reshape(Y.shape[0])
#n_input = 50
#x_train = PCA(n_components = n_input).fit_transform(X)
#y_train = Y
x_train = MinMaxScaler().fit_transform(x_train)
x_test = MinMaxScaler().fit_transform(x_test)
X = x_train



# DEFINE HYPERPARAMETERS
learning_rate   = 0.0001
training_epochs = 100
mini_batch_size = 100           # mini_batch_size = X.shape[0]-1
display_step    = 10             # how often to display loss and accuracy
num_hidden_1    = 30*32             # 1st hidden layer num features
num_hidden_2    = 30*8             # 2nd hidden layer num features
num_bottleneck  = 30              # bottleneck num features
num_input       = X.shape[1]     # scRANAseq data input (number of genes)
dropout_prob    = 0.2
l2_parameter    = 1e-4

initializer = tf.compat.v1.initializers.glorot_normal(seed=None)
bias_initializer = tf.keras.initializers.Zeros()

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

# CONSTRUCT AUTOENCODER MODEL
print("\n" + "Constructing Autoencoder Model ..." + "\n")

# Zero-inflated negative binomial (ZINB) model is for modeling count variables with excessive zeros and it is usually for overdispersed count outcome variables.
def zinb_model(x, mean, inverse_dispersion, logit, eps=1e-8): 
                                      
    expr_non_zero = - tf.nn.softplus(- logit) \
                    + tf.log(inverse_dispersion + eps) * inverse_dispersion \
                    - tf.log(inverse_dispersion + mean + eps) * inverse_dispersion \
                    - x * tf.log(inverse_dispersion + mean + eps) \
                    + x * tf.log(mean + eps) \
                    - tf.lgamma(x + 1) \
                    + tf.lgamma(x + inverse_dispersion) \
                    - tf.lgamma(inverse_dispersion) \
                    - logit 

    expr_zero = - tf.nn.softplus( - logit) \
                + tf.nn.softplus(- logit + tf.log(inverse_dispersion + eps) * inverse_dispersion \
                                 - tf.log(inverse_dispersion + mean + eps) * inverse_dispersion) 
    
    template = tf.cast(tf.less(x, eps), tf.float32)
    expr =  tf.multiply(template, expr_zero) + tf.multiply(1 - template, expr_non_zero)
    return tf.reduce_sum(expr, axis=-1)

def model_and_loss():
    input_layer = Input(shape=(num_input,))
    layer_1 = Dense(num_hidden_1,activation='relu',kernel_initializer=initializer, bias_initializer= bias_initializer)(input_layer)
    layer_2 = Dense(num_hidden_2,activation='relu',kernel_initializer=initializer, bias_initializer= bias_initializer)(layer_1)
    bottleneck = Dense(num_bottleneck, kernel_initializer=initializer, bias_initializer= bias_initializer)(layer_2)
    layer__2 = Dense(num_hidden_2, activation='relu',kernel_initializer=initializer, bias_initializer= bias_initializer)(bottleneck)
    layer__1 = Dense(num_hidden_1, activation='relu',kernel_initializer=initializer, bias_initializer= bias_initializer)(layer__2)
    #output = Dense(num_input)(layer__1)
    pi = Dense(num_input)(layer__1)
    mean = Dense(num_input, activation=MeanAct)(layer__1)
    disp = Dense(num_input, activation=DispAct)(layer__1)
    
    zinb = ZINB(pi, theta=disp)
    
    ae_model = Model(input_layer, outputs=mean)
    encoder_model = Model(input_layer, outputs=bottleneck)

    return ae_model, zinb, encoder_model
    #return ae_model


def model_1_and_loss():
    input_layer = Input(shape=(num_input,))
    layer_1 = Dense(num_hidden_1,activation='relu',kernel_initializer=initializer, bias_initializer= bias_initializer, kernel_regularizer=regularizers.l2(l2_parameter))(input_layer)
    layer_1 = Dropout(rate=dropout_prob)(layer_1)
    bottleneck = Dense(num_bottleneck,  activation='relu', kernel_initializer=initializer, bias_initializer= bias_initializer)(layer_1)
    layer__1 = Dense(num_hidden_1, activation='relu',kernel_initializer=initializer, bias_initializer= bias_initializer, kernel_regularizer=regularizers.l2(l2_parameter))(bottleneck)
    layer__1 = Dropout(rate=dropout_prob)(layer__1)
    #output = Dense(num_input)(layer__1)
    pi = Dense(num_input, kernel_initializer=initializer, activation='sigmoid')(layer__1)
    mean = Dense(num_input, kernel_initializer=initializer, activation=MeanAct)(layer__1)
    disp = Dense(num_input, kernel_initializer=initializer, activation=DispAct)(layer__1)
    
    zinb = ZINB(pi, theta=disp)
    #zinb_loss = zinb_model(input_layer, mean, disp, pi)
    
    ae_model = Model(input_layer, outputs=mean)
    encoder_model = Model(input_layer, outputs=bottleneck)

    return ae_model, zinb, encoder_model

def mse_loss(y_true, y_pred):
    cost = tf.pow(y_true - y_pred, 2)
    return cost

autoencoder, zinb, encoder = model_1_and_loss()
#autoencoder = model_and_loss()
#cost = tf.pow(y_true - y_pred, 2)

autoencoder.compile(optimizer='adam', loss=zinb.loss)
history = autoencoder.fit(x_train, x_train, 
                epochs=training_epochs,
                batch_size=mini_batch_size,
                shuffle=True,
                validation_data=(x_test,x_test))


#Plot the loss function
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("ZINB Autoencoder 6 layer Loss with L2 and Dropout Regularization (" + str(training_epochs) + " epochs)")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



#Plot the TSNE for the training and see how the tissues cluster
y_pred=encoder.predict(x_train)
model_tsne_auto = TSNE(learning_rate = 100, n_components = 2, random_state = 123,
                       perplexity = 90, n_iter = 1000, verbose = 1)
tsne_auto = model_tsne_auto.fit_transform(y_pred)

expr_auto={}
expr_auto['tissue'] = y_train
expr_auto['tsne_1'] = tsne_auto[:,0]
expr_auto['tsne_2'] = tsne_auto[:,1]
plt.figure(figsize=(10,8))
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="tissue",
    palette=sns.color_palette("bright", 5),
    data=expr_auto,
    legend="full",
    alpha=0.3
)
ax = plt.gca()
ax.set_title("tsne on ZINB Autoencoder 6 layer with L2 and Dropout Regularization (" + str(training_epochs) + " epochs)")

    
    

    

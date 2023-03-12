import params as par
import tensorflow as tf
import train
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

class Model(object):
    def __init__(self,W,b) -> None:
        self.W = W
        self.b = b
    def call(self,x):
        z = tf.add(self.b,tf.matmul(x,self.W))
        y = tf.sigmoid(z)
        return y
    def train(self, training_data,labels,epochs = par.NUM_EPOCHS,lr = par.LEARNING_RATE):
        train.train(self, training_data,labels,epochs = par.NUM_EPOCHS,lr = par.LEARNING_RATE)

def model_def():
    if(par.MODEL_TYPE=="feed-forward"):
        """ model = Sequential()
        output_layer = Dense(par.DIM_OUTPUT, par.ACTIVATION)
        model.add(output_layer)
        #model.build(input_shape = [par.DIM_INPUT,])
        #model.save(par.MODEL_PATH) """

        W = tf.Variable(tf.zeros([par.DIM_INPUT,par.DIM_OUTPUT],dtype = tf.float64),shape=[par.DIM_INPUT,par.DIM_OUTPUT])
        b = tf.Variable(tf.zeros([1,par.DIM_OUTPUT],dtype = tf.float64),shape = [1,par.DIM_OUTPUT])
        model = Model(W,b)

    return model
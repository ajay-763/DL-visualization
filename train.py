import params as par
import numpy as np
import tensorflow as tf

def loss(out,out_actual):
    return np.sum(np.square((out_actual - out)))

def train(model, training_data,labels,epochs = par.NUM_EPOCHS,lr = par.LEARNING_RATE):
    for epoch in range(epochs):
        for i in range(par.NUM_INPUTS):
            input = training_data[i:i+1,:]
            label = tf.constant([[labels[i]],],shape = [1,1])
            #label = [[labels[i]],]
            with tf.GradientTape() as gW:
                with tf.GradientTape() as gb:
                    gW.watch(model.W)
                    gb.watch(model.b)
                    out = model.call(input)
                    """ mse = tf.keras.losses.MeanSquaredError()
                    l = mse(label, out) """
                    bce=tf.keras.losses.BinaryCrossentropy()
                    l = bce(label,out)
            gradW = gW.gradient(l,model.W)
            gradb = gb.gradient(l,model.b)
            model.W = model.W - lr*gradW
            model.b = model.b - lr*gradb 
            print("Epoch = ",epoch,"index = ",i,"Loss = ",(l.numpy()))
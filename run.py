import pandas as pd
import numpy as np
import params as par
import model_def
import tensorflow as tf
import plot
model = model_def.model_def()

from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

training_df = pd.read_csv(par.TRAINING_PATH)
training_inputs = np.zeros([par.NUM_INPUTS, par.DIM_INPUT],dtype=np.float64)
training_inputs[:,0] = training_df['in1']
training_inputs[:,1] = training_df['in2']
training_inputs = tf.constant(training_inputs,shape = [par.NUM_INPUTS,par.DIM_INPUT],dtype = tf.float64)
labels = training_df['label']
input_df = pd.read_csv(par.INPUT_PATH)
input = np.zeros([1,par.DIM_INPUT])
input[0,0] = input_df['in1']
input[0,1] = input_df['in2']
input = tf.constant(input,dtype=tf.float64,shape = [1,par.DIM_INPUT])

model.train(training_inputs,labels)

output = model.call(input)
output_df = {'label':(output.numpy())[0]}
output_df = pd.DataFrame.from_dict(output_df)
output_df.to_csv(par.OUTPUT_PATH)

plot.plot(model,training_df)
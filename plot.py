import matplotlib.pyplot as plt
import params as par
import tensorflow as tf

def plot(model,training_df):
    input1_label0 = []
    input1_label1 = []
    input2_label0 = []
    input2_label1 = []

    for i in range(28):
        for j in range(28):
            in1 = 0.25*i
            in2 = 0.25*j
            input = tf.constant([in1,in2],dtype = tf.float64, shape = [1,2])
            output = ((model.call(input)).numpy())[0]
            if(output<0.5):
                input1_label0.append(in1)
                input2_label0.append(in2)
            else:
                input1_label1.append(in1)
                input2_label1.append(in2)
    
    plt.figure('Trained model')
    plt.scatter(input1_label0,input2_label0,marker = "o")
    plt.scatter(input1_label1,input2_label1,marker = "^")
    plt.xlim((0,7))
    plt.ylim((0,7))
    
    input1_label0 = []
    input1_label1 = []
    input2_label0 = []
    input2_label1 = []

    labels = training_df['label']
    inp1 = training_df['in1']
    inp2 = training_df['in2']
    for i in range(par.NUM_INPUTS):
        if(labels[i]==0):
            input1_label0.append(inp1[i])
            input2_label0.append(inp2[i])
        else:
            input1_label1.append(inp1[i])
            input2_label1.append(inp2[i])
    
    plt.figure('Training data')
    plt.scatter(input1_label0,input2_label0,marker = "+")
    plt.scatter(input1_label1,input2_label1,marker = "x")
    plt.xlim((0,7))
    plt.ylim((0,7))

    plt.show()
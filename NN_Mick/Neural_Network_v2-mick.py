import pandas as pd
import numpy as np
print('hhhh')


########################Data:###########################
data=pd.read_csv('Data.csv')
data=data.sample(frac=1).reset_index(drop=True)

train_data=data.head(12)    # head goes top down
test_data=data.tail(4)  # tail goes bottom up

train_input_data=train_data.drop(['O'],axis=1)  # input data = remove the output cols
print(type(train_input_data))
train_input=train_input_data.values # remove the headers. TURN INTO NP ARRAY
print(type(train_input))

train_output_data=train_data.drop(['A','B','C','D'],axis=1)
train_output=train_output_data.values

test_input_data=test_data.drop(['O'],axis=1)
test_input=test_input_data.values
test_output_data=test_data.drop(['A','B','C','D'],axis=1)
test_output=test_output_data.values
print('test o: ', test_output)
#######################Training##############################

# 4x1 column vector! 4 rows 1 column
weights=np.matrix(2*np.random.random((4,)) -1).T

print('input 0',' ',weights)


loop=1000

for q in range(loop):
    prediction=[]
    
    # loop through all inputs
    for i in range(0,len(train_input)):
    
        # multiply weight matrix by input matrix. dot product does the weighted sum function
        # this is for a single neuron, but scales to all neurons if we use matrix multiplication not dot product
        predic=1/(1 + (np.exp(-(np.dot(np.matrix(train_input[i]),weights)))))
        # single element of error vector
        error=predic-train_output[i]
        # some method to update weights?
        error_activ=error*(predic*(1-predic))
        adjest_weights=np.dot(np.matrix(train_input[i]).T,error_activ)
        weights += adjest_weights
        #print('input ',i+1,' ',weights)
        prediction.append(predic)
        #print(predic,'-',train_output[i],'=',error)
        
    if q==100:
        print(np.array(prediction))
        print(train_output)
        
#####################Test########################################
loop=50

for q in range(loop):
    prediction=[]
    for i in range(0,len(test_input)):
        
        predic=1/(1 + (np.exp(-(np.dot(np.matrix(test_input[i]),weights)))))
        print('predict:', predic, 'vs actual: ', test_output[i])
        


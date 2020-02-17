import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint


def activ_fn(x):
    return 1 / (1 + np.exp(-x))

def create_wmatrix(struct):
    w = []
    for i in range(len(struct) - 1):
        cols_rows = (struct[i + 1], struct[i])
        rand_weights = 2 * np.random.random(cols_rows) - 1
        w.append(rand_weights)
        
    return w


def runNN(inputs, struct, w):
    
    # COLUMN VECTOR. also make it a matrix ndim2
    inputs = np.array(inputs, ndmin=2).T
    
    # LOOP HIDDEN LAYERS + OUTPUT
    outputs = None
    for i in range(len(struct) - 1):
        outputs = activ_fn(np.dot(w[i], inputs))
        inputs = outputs
        
    return outputs

def populate(size):
    
    pop = [None] * size
    for i in range(len(w)):
        individual_weights = []
        for k in range(len(struct) - 1):
            cols_rows = (struct[k + 1], struct[k])
            rand_weights = 2 * np.random.random(cols_rows) - 1
            individual_weights.append(rand_weights)
        pop[i] = individual_weights
    
    return pop

def cycle(pop, target, size, generations):

    pop = populate(size)
    
    for gen in range(generations):
        gen += 1
        population,cord,top_score,answers=fitness(population,target,size)
        #print('fitness:',len(population))
        
        print('Generation:',gen)
        x=float(np.round(top_score[0],20))
        print('fittest: ',x)
    
        population=mate(population,size)
        #print('mate:',len(population))
        
        if x <= 0:
            break


def fitness(pop, target, size):
    scores = [] 
    pop = []
    cord = []
    for index, individual in enumerate(pop):
        predict_outs = runNN(individual)
        error=mse(answers)
        
        score=np.mean(error)-target
        
        scores.append((score,entity,index))
        cord.append((score,score))
        #pop.append((entity,index))
    print('done')
    #print('scores:',len(scores),type(scores))#,np.array(scores).shape)
    #top_score=sorted(scores)[:size]
    top_score=np.array(scores)[np.array(scores)[:,0].argsort()]
    #print('top_score:',len(top_score),type(top_score),top_score[0][0])
    for i in range(size):
        order=top_score[i][1]
        pop.append(order)
    
    return np.array(pop)[:int(size*0.1)],np.array(cord),top_score[0],answers


data = pd.read_csv(r'Data.csv')
data = data.sample(frac=1).reset_index(drop=True)

train_input_data = data.drop(['O'], axis = 1)
train_input = np.matrix(train_input_data.values)
train_output_data = data.drop(['A','B','C','D'], axis = 1)
train_output = np.matrix(train_output_data.values)




size = 2
struct = [3, 10, 10, 4]   
struct = [2, 3, 2]   
#inputs = [1, 2, 3]
#w = create_wmatrix(struct)

#outputs = runNN(inputs, struct, w)
#print('Initial Guess:', outputs)



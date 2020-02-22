import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint


# GLOBAL ATM, CHANGE TO PARAM
def mse(predict_outs):
    
    error = []
    for i in range(len(train_output)):
        mse = (train_output[i] - answers[i]) ** 2
        error.append(mse)
        
    error = np.mean(np.array(error))
    return error

def activ_fn(x):
    return 1 / (1 + np.exp(-x))

# returns a list, each item is a weight matrix (np array)
def create_wmatrix(struct):
    w = []
    for i in range(len(struct) - 1):
        # create the weights according to NN struct (no of nodes!)
        cols_rows = (struct[i + 1], struct[i])
        rand_weights = 2 * np.random.random(cols_rows) - 1
        w.append(rand_weights)
    return w


def runNNLoop(inputs, struct, w):
    outputs = []
    for i in range(len(inputs)):
        single_output = runNN(inputs[i], struct, w)
        outputs.append(single_output)
    return outputs
   
def runNN(inputs, struct, w):
    
    # COLUMN VECTOR. also make it a matrix ndim2
    # have checks here on data coming in??
    # no of inputs should match struct - put error code here
    #inputs = np.array(inputs, ndmin=2).T
    
    # LOOP HIDDEN LAYERS + OUTPUT
    outputs = None
    for i in range(len(struct) - 1):
        outputs = activ_fn(np.dot(w[i], inputs))
        inputs = outputs
        
    return outputs

def populate(size, struct):
    
    # each element in list contains the NN weights for each snake individual
    # subelements = hidden layer weight matrices
    # they have the correct dimensions
    # --maybe just use create_wmatrix() here instead of nested for loops
    population = [None] * size
    for i in range(len(population)):
        individual_weights = create_wmatrix(struct)
        population[i] = individual_weights
    
    return population


# mate 2 individuals - cross the weights together
# p1 = entire weight matrix for one individual
# p1[0] = wi and p1[1] = wh and p1[2] = w0. each matrix is a 'gene'
# we should only cross genes of matching layers???
# this would be 'mixing' genes instead of swapping genes
# or define genes inside the weight matrix
def mate(population, pop_size, struct):
    #print('mate')
    # dont need to pass in popsize as we can calculate it using population
    pop_new=[None]*pop_size
    index1 = 0
    index2 = 1
    
    # pair 2 individuals, so loop popsize/2 times
    for _ in range(int(pop_size/2)):
        
        # pick random indivs (not based on fittest!)
        # or has pop been reordered so fittest are in first 50%?
        p1=population[np.random.randint(0, len(population))]
        p2=population[np.random.randint(0, len(population))]
        
        # one-point crossover (first half of w_matrices are swapped)
        # TWO CHILDREN
        for w_matrix in range(len(p1) / 2):
            p1[w_matrix], p2[w_matrix] = p2[w_matrix], p1[w_matrix]
            
        # total mix crossover (mix elements of every w_matrix)
        # if ele are the same they dont change
        # ONE CHILD? this menas pop size is reduced in half though...
        #for w_matrix in range(len(p1)):
         #   for ele in w_matrix:
          #      p1[w_matrix][ele] = (p1[w_matrix][ele] + p2[w_matrix][ele]) / 2
            
        # MUTATE - pick a random gene (w_matrix)
        # generate a new weight matrix completely randomly
        # or just modify the current one along a gaussian +/- 
        # need to know the index to put it back in p1
        # or pass in struct
        #struct = range(len(p1[0]))
        #random_gene = np.random.random()
        #random_gene = random.choice(p1)

        # could gen range(len()) myself then pick random from that list / iterator
        # but randrange does this 
        
        # P1
        random_gene_index = randrange(len(p1))
        random_gene = p1[random_gene_index]
        mutation=2*np.random.random(random_gene.shape)-1
        mutated_gene = random_gene + mutation
        p1[random_gene_index] = mutated_gene
        
        # P2
        random_gene_index = randrange(len(p2))
        random_gene = p2[random_gene_index]
        mutation=2*np.random.random(random_gene.shape)-1
        mutated_gene = random_gene + mutation
        p2[random_gene_index] = mutated_gene
        
        pop_new[index1] = p1
        pop_new[index2] = p2
        index1 += 2
        index2 += 2
    return 


def genLoop(population, target, size, generations, struct):
    
    # INITIAL POPULATION WITH RANDOM WEIGHTS
    population = populate(size, struct)
    
    for gen in range(generations):
        gen += 1
        population, cord ,top_fitness, answers = popLoop(population, target, size, struct)
        #print('fitness:',len(population))
        
        print('Generation:', gen)
        x = float(np.round(top_fitness[0], 20))
        print('fittest: ', x)
    
        population = mate(population, size)
        #print('mate:',len(population))
        
        # Termination = best fitness = zero error!
        if x <= 0:
            break


def popLoop(population, target, size, struct):

    # numpy slicing modifies orig array, so we make a duplicate to hold the sorted data!
    population_complete = []
    fitness_list = [] 
    cord = []
    
    # loop through each snake individual
    # calculate fitness / score
    for index, individual in enumerate(population):
        predict_outs = gameLoop(individual, struct)
        error = mse(predict_outs)
        fitness = error - target
        
        # tuple for each individual
        fitness_list.append((fitness, individual, index))
        cord.append((fitness, fitness))
        
    # grab the fitnesses, sort by lowest then return the matching index
    # [:, 0] is numpy slicing, don't understand syntax
    top_index = np.array(fitness_list)[:, 0].argsort()
    top_fitness = np.array(fitness_list)[top_index]
    
    # re-order??
    for i in range(size):
        order = top_fitness[i][1]
        population_complete.append(order)
    
    # return top 10% individuals
    population_complete = np.array(population_complete)[:int(size*0.1)]
    return population_complete, np.array(cord), top_fitness[0], answers

def gameLoop(individual, struct):
    # individual = the weights for NN
    predict_outs = runNNLoop(train_input, struct, individual)
    return predict_outs

def draw(cord, target):
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.scatter(cord[:,0],cord[:,1],c='green',s=12)
    plt.scatter(target[0], target[1], c ='red', s = 60)

data = pd.read_csv(r'Data.csv')
data = data.sample(frac=1).reset_index(drop=True)

train_input_data = data.drop(['O'], axis = 1)
train_input = np.matrix(train_input_data.values)

train_output_data = data.drop(['A','B','C','D'], axis = 1)
train_output = np.matrix(train_output_data.values)
#actual_outs = train_output



size = 2
struct = [3, 10, 10, 4]   
struct = [2, 3, 2]  
#inputs = [1, 2, 3]
#w = create_wmatrix(struct)

#outputs = runNN(inputs, struct, w)
#print('Initial Guess:', outputs)

target = 0
size = 5
generations = 100
population = []

#population = populate(size, struct)
#pprint.pprint(population)

population, cord, answers = genLoop(population, target, size, generations, struct)
draw(cord,(target,target))
for i in range(len(answers)):
    print(np.round(answers[i],2),'===',train_output[i])



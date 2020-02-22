import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pprint
import math


# GLOBAL ATM, CHANGE TO PARAM
# this only works for single output - based on our data - change this?
def mse(predict_outs):
    # print('\n\n\npredict', predict_outs)
    # print('actual', train_output)

    error = []
    for i in range(len(train_output)):
        # print(train_output[i])
        mse = (train_output[i] - predict_outs[i]) ** 2
        error.append(mse)
            
    error = np.mean(np.array(error))
    return error

def activ_fn(x):
    return 1 / (1 + np.exp(-x))

# returns a list, each item is a weight matrix (np array)
def create_wmatrix(struct):
    weights = []
    for i in range(len(struct) - 1):
        # create the weights according to NN struct (no of nodes!)
        cols_rows = (struct[i + 1], struct[i])
        rand_weights = 2 * np.random.random(cols_rows) - 1
        # print('\n\nw shape:', rand_weights.shape)
        rand_weights = rand_weights.T
        weights.append(rand_weights)
    return weights


def runNNLoop(inputs, struct, weights):
    # inputs = total array of column vectors
    outputs = []
    for i in range(len(inputs)):
        single_output = runNN(inputs[i], struct, weights)
        outputs.append(single_output)
    # print(outputs, '\n')
    return outputs
   
def runNN(inputs, struct, weights):
    # inputs is a single column vector
    # (saved as a row in csv so change to col using .T)
    # print('\n\nbefore T', inputs.shape)
    # inputs = inputs.T
    # inputs = np.matrix(inputs).T
    # print('after T', inputs.shape)
    #print(inputs)

    
    # COLUMN VECTOR. also make it a matrix ndim2
    # have checks here on data coming in??
    # no of inputs should match struct - put error code here
    #inputs = np.array(inputs, ndmin=2).T
    
    # LOOP HIDDEN LAYERS + OUTPUT
    outputs = None
    for i in range(len(struct) - 1):
        # outputs = activ_fn(np.dot(weights[i], inputs))
        # for reyan version without transpose. np.dot left to right matters!
        outputs = activ_fn(np.dot(inputs, weights[i]))
        inputs = outputs
        
    # each output is np.matrix though it only says this when its in bigger list
    #print(outputs, '\n')
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
        # print('\n\nw shape:', np.array(individual_weights).shape)

    return population


# mate 2 individuals - cross the weights together
# p1 = entire weight matrix for one individual
# p1[0] = wi and p1[1] = wh and p1[2] = w0. each matrix is a 'gene'
# we should only cross genes of matching layers???
# this would be 'mixing' genes instead of swapping genes
# or define genes inside the weight matrix

#param name = pop_sorted here?
def mate(population, pop_size):
    #print('mate')
    # dont need to pass in popsize as we can calculate it using population
    pop_new=[None]*pop_size
    index1 = 0
    index2 = 1
    # print(len(population))
    
    # pair 2 individuals, so loop popsize/2 times
    for _ in range(int(pop_size/2)):
        
        # pick random indivs (not based on fittest!)
        # or has pop been reordered so fittest are in first 50%?
        p1=population[np.random.randint(0, len(population))]
        p2=population[np.random.randint(0, len(population))]
        # print('\n\np2', p2)
##############
        p1_flat = []
        p2_flat = []
        #
        # for w_matrix in p1:
        #     flat = w_matrix.flatten()
        #     p1_flat = np.concatenate((p1_flat, flat))
        # # print(total)
        # for w_matrix in p2:
        #     # print(w_matrix.shape)
        #     flat = w_matrix.flatten()
        #     p2_flat = np.concatenate((p2_flat, flat))
        #
        # # print(p2_flat.shape)
        # child1 = np.concatenate(([p1_flat[:10],p2_flat[10:]]))
        # child2 = np.concatenate(([p2_flat[:10], p1_flat[10:]]))
        #
        # mutation1 = 2 * np.random.random((20,)) - 1
        # mutation2 = 2 * np.random.random((20,)) - 1
        #
        # child1 = child1 + mutation1
        # child2 = child2 + mutation2
        #
        # # wrap back into my form
        # index = 0
        # for i in range(len(p1) - 1):
        #     # create the weights according to NN struct (no of nodes!)
        #     cols_rows = (struct[i + 1]* struct[i]) + index
        #     cols_rows2 = (struct[i + 1], struct[i])
        #
        #     # use opposite of flattern here!
        #     p1[i] = child1[index:cols_rows].reshape(cols_rows2)
        #     index = cols_rows + 1
        #
        # index = 0
        # for i in range(len(p2) - 1):
        #     # create the weights according to NN struct (no of nodes!)
        #     cols_rows = (struct[i + 1]* struct[i]) + index
        #     cols_rows2 = (struct[i + 1], struct[i])
        #
        #     # use opposite of flatten here!
        #     p2[i] = child2[index:cols_rows].reshape(cols_rows2)
        #     index = cols_rows + 1
##############
        #print('\n\np1', p1)

        # one-point crossover (first half of w_matrices are swapped)
        # TWO CHILDREN
        # print('\n\np1 before', p1)
        for w_matrix in range(math.ceil(len(p1) / 2)):
            p1[w_matrix], p2[w_matrix] = p2[w_matrix], p1[w_matrix]
        # print('p1 after', p1)
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

        # add strength parameter (-2 to 2)
        # rate = no of weights affected

        # P1
        #random.choice here instead
        #for loop = adjust mutation rate of chromosome
        # for i in range(len(p1)):
        #     mutation = 2 * np.random.random(p1[i].shape) - 1
        #     p1[i] += mutation
        #     mutation = 2 * np.random.random(p2[i].shape) - 1
        #     p2[i] += mutation

        # random_gene_index = random.randrange(len(p1))
        # random_gene = p1[random_gene_index]
        # # print('\nrandom', random_gene)
        # mutation=2*np.random.random(random_gene.shape)-1
        # mutated_gene = random_gene + mutation
        # # print('mutate', mutated_gene)
        # p1[random_gene_index] = mutated_gene
        #
        # # P2
        # random_gene_index = random.randrange(len(p2))
        # random_gene = p2[random_gene_index]
        # mutation=2*np.random.random(random_gene.shape)-1
        # mutated_gene = random_gene + mutation
        # p2[random_gene_index] = mutated_gene
        
        pop_new[index1] = p1
        pop_new[index2] = p2
        index1 += 2
        index2 += 2
    # print(pop_new)
    del pop_new[-1]
    return pop_new


def genLoop(population, target, size, generations, struct):
    
    # INITIAL POPULATION WITH RANDOM WEIGHTS
    population = populate(size, struct)
    # print(population[0])
    
    for gen in range(generations):
        gen += 1
        population, top_fitness = popLoop(population, target, size, struct)
        # print(population_sorted)
        #print('fitness:',len(population_sorted))
        
        print('Generation:', gen)
        x = float(np.round(top_fitness, 20))
        print('fittest: ', x)

        population = mate(population, size)

        # print(population_sorted)
        #print('mate:',len(population))
        
        # Termination = best fitness = zero error!
        if x <= 0:
            break


def popLoop(population, target, pop_size, struct):

    # numpy slicing modifies orig array, so we make a duplicate to hold the sorted data!
    population_sorted = []
    fitness_list = []
    individual_list = []
    index_list = []
    cord = []
    
    # loop through each snake individual
    # calculate fitness / score
    for index, individual in enumerate(population):
        predict_outs = gameLoop(individual, struct)
        error = mse(predict_outs)
        # print(error)
        fitness = error - target
        # print(fitness)

        # COME BACK TO THIS LATER, SORT NUMPY ARRAY PROPERLY
        # tuple for each individual
        #fitness_list.append((fitness, individual, index))
        # print(fitness_list)

        fitness_list.append(fitness)
        individual_list.append(individual)
        index_list.append(index)

        cord.append((fitness, fitness))
        # print(index)

    #top_score=np.array(fitness_list)[np.array(fitness_list)[:,0].argsort()]

    # sort by best fitness
    # rename these
    # indexes are all that matter here
    sorted_fitness_list = np.sort(fitness_list)
    sorted_fitness_index = np.array(fitness_list).argsort()
    # print(fitness_list)
    # print(sorted_fitness_list)
    # print(sorted_fitness_index)

    for index in sorted_fitness_index:
        # population_sorted.append(individual_list[index])
        population_sorted.append(population[index])

    # predict_outs = gameLoop(population_sorted[0], struct)
    # error = mse(predict_outs)
    # fitness = error - target
    # print(fitness, sorted_fitness_list[0])

    # print(len(population_sorted))

    # return top 10% individuals + best fitness
    top10 = math.ceil(pop_size / 10)
    print(len(population_sorted))
    population_sorted = population_sorted[0:top10]
    print(len(population_sorted))
    return population_sorted, sorted_fitness_list[0]

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

# matrix will transpose it
train_input_data = data.drop(['O'], axis = 1)
train_input = np.matrix(train_input_data.values)

train_output_data = data.drop(['A','B','C','D'], axis = 1)
train_output = np.matrix(train_output_data.values)
# print(train_output)
#actual_outs = train_output


# struct has to match the input and output data
size = 1000
# struct = [3, 10, 10, 4]
struct = [4, 3, 2, 1]
#inputs = [1, 2, 3]
#w = create_wmatrix(struct)

#outputs = runNN(inputs, struct, w)
#print('Initial Guess:', outputs)

target = 0
generations = 100
population = []

#population = populate(size, struct)
#pprint.pprint(population)

# genloop = cycle
# population, cord, answers = genLoop(population, target, size, generations, struct)
genLoop(population, target, size, generations, struct)
# draw(cord,(target,target))
# for i in range(len(answers)):
#     print(np.round(answers[i],2),'===',train_output[i])



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pprint
import math
import copy


def mse(predict_output, train_output):
    error = []
    for i in range(len(train_output)):
        for k in range(len(train_output[i])):
            mse = (train_output[i][k] - predict_output[i][k]) ** 2
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
        # here the random weights can sometimes be around the same... use diff function?
        # (row, col) form used, if doing opposite then apply .T operation elsewhere
        rows_cols = (struct[i], struct[i + 1])
        rand_weights = 2 * np.random.random(rows_cols) - 1
        # rand_weights = rand_weights.T
        weights.append(rand_weights)
    return weights


def runNNLoop(inputs, struct, weights):
    # inputs = total array of input vectors
    outputs = []
    for i in range(len(inputs)):
        single_output = runNN(inputs[i], struct, weights)
        outputs.append(single_output)
    return outputs


def runNN(inputs, struct, weights):
    # inputs is a single row vector
    # (saved as a row in csv)

    # LOOP HIDDEN LAYERS + OUTPUT
    # matrix multiplication happens here!
    outputs = None
    for i in range(len(struct) - 1):
        outputs = activ_fn(np.dot(inputs, weights[i]))
        inputs = outputs
    return outputs


def populate(size, struct):
    # each element in list contains the NN weights for each snake individual
    # subelements = hidden layer weight matrices, built up using correct dimensions
    # this form can be flattened to a single chromosome (used for crossover)
    population = [None] * size
    for i in range(len(population)):
        individual_weights = create_wmatrix(struct)
        population[i] = individual_weights
    return population


# mate 2 individuals - cross the weights together
# p1 = entire weight matrix for one individual
def mate(population, pop_size):
    pop_new = []

    # pair 2 individuals, so loop popsize/2 times
    for _ in range(int(pop_size / 2)):

        # pick random snakes (not based on fittest!)
        # chance to pick the same 2
        p1 = population[np.random.randint(0, len(population))]
        p2 = population[np.random.randint(0, len(population))]
        p1_flat = []
        p2_flat = []

        # flatten into 1D chromosome
        for w_matrix in p1:
            flat = w_matrix.flatten()
            p1_flat = np.concatenate((p1_flat, flat))

        for w_matrix in p2:
            flat = w_matrix.flatten()
            p2_flat = np.concatenate((p2_flat, flat))

        # single point crossover using half-length
        half_length = math.ceil(len(p1_flat) / 2)
        child1 = np.concatenate(([p1_flat[:half_length], p2_flat[half_length:]]))
        child2 = np.concatenate(([p2_flat[:half_length], p1_flat[half_length:]]))

        # this shows that crossover has no effect really!
        # child1 = copy.deepcopy(p1_flat)
        # child2 = copy.deepcopy(p2_flat)

        # mutate every single gene/weight
        mutation1 = 2 * np.random.random((len(p1_flat),)) - 1
        mutation2 = 2 * np.random.random((len(p1_flat),)) - 1
        child1 = child1 + mutation1
        child2 = child2 + mutation2

        # wrap back into my form
        p1_new = []
        p2_new = []
        index1 = 0
        for w_matrix in p1:
            rows, cols = w_matrix.shape
            index2 = index1 + (rows * cols)
            p1_new.append(np.reshape(np.array(child1[index1:index2]), (rows, cols)))
            p2_new.append(np.reshape(np.array(child2[index1:index2]), (rows, cols)))
            index1 = index2

        pop_new.append(p1_new)
        pop_new.append(p2_new)

    return pop_new


def genLoop(population, target, size, generations, struct, train_output):
    # INITIAL POPULATION WITH RANDOM WEIGHTS
    population = populate(size, struct)

    for gen in range(generations):
        gen += 1

        # calc fitness for entire pop. select only top 10%
        population, top_fitness = popLoop(population, target, size, struct, train_output)

        print('Generation:', gen)
        x = float(np.round(top_fitness, 20))
        print('fittest: ', x)

        # mate the top 10% to produce original size population
        population = mate(population, size)

        # Termination = best fitness = zero error!
        if x <= 1e-10:
            break


def popLoop(population, target, pop_size, struct, train_output):
    # numpy slicing modifies orig array, so we make a duplicate to hold the sorted data!
    population_sorted = []
    fitness_list = []
    individual_list = []
    index_list = []
    cord = []

    for index, individual in enumerate(population):
        # run the gameLoop for a single snake
        predict_output = gameLoop(individual, struct)
        error = mse(predict_output, train_output)
        fitness = error

        fitness_list.append(fitness)
        individual_list.append(individual)
        index_list.append(index)
        cord.append((fitness, fitness))

    # sort by best fitness - grab the indexes and put back into original population
    sorted_fitness_list = np.sort(fitness_list)
    sorted_fitness_index = np.array(fitness_list).argsort()
    top_fitness = sorted_fitness_list[0]


    for index in sorted_fitness_index:
        population_sorted.append(population[index])

    # return top 10% individuals + top fitness
    top10 = math.ceil(pop_size * 0.1)
    population_sorted = population_sorted[0:top10]

    return population_sorted, top_fitness


def gameLoop(individual, struct):
    predict_output = runNNLoop(train_input, struct, individual)
    return predict_output


def draw(cord, target):
    plot.xlim((-1, 1))
    plot.ylim((-1, 1))
    plot.scatter(cord[:, 0], cord[:, 1], c='green', s=12)
    plot.scatter(target[0], target[1], c='red', s=60)



# NEED TO SPLIT DATA HERE WITH TRAINING AND VALIDATION....
data = pd.read_csv(r'Data.csv')
# data = pd.read_csv(r'Data2.csv')
data = data.sample(frac=1).reset_index(drop=True)

# matrix will transpose it and also use [[values]] whereas array has [values]
# train_input_data = data.drop(['O1', 'O2'], axis = 1)
train_input_data = data.drop(['O'], axis = 1)
train_input = np.array(train_input_data.values)

train_output_data = data.drop(['A','B','C','D'], axis = 1)
train_output = np.array(train_output_data.values)
# print(train_output)
#actual_outs = train_output


# struct has to match the input and output data
# very simple dataset so only one hidden layer works well
size = 100
# struct = [3, 10, 10, 4]
struct = [4, 8, 2]
# struct = [4, 8, 8, 8, 8, 4, 1]
#inputs = [1, 2, 3]
#w = create_wmatrix(struct)

#outputs = runNN(inputs, struct, w)
#print('Initial Guess:', outputs)

target = 0
generations = 200
population = []

#population = populate(size, struct)
#pprint.pprint(population)

# genloop = cycle
# population, cord, answers = genLoop(population, target, size, generations, struct)
genLoop(population, target, size, generations, struct, train_output)
# draw(cord,(target,target))
# for i in range(len(answers)):
#     print(np.round(answers[i],2),'===',train_output[i])




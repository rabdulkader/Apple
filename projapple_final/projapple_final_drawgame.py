import pygame, time, random, copy, math
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plot
from scipy.special import expit as activ_fn
import pandas as pd

# GLOBALS
white = [255, 255, 255]
black = [0, 0, 0]
red = [255, 0, 0]
blue = [0, 0, 255]
green = [0, 255, 0]
grey = [220, 220, 220]
purple = [128, 0, 128]
orange = [255, 128, 0]



def updateVision(distWall, snake, step_size, dirDict, width, height, apple):
    # WALL DISTANCE
    distWall["N"] = abs(0 - snake[0][1])
    distWall["S"] = abs((height - step_size) - snake[0][1])
    distWall["W"] = abs(0 - snake[0][0])
    distWall["E"] = abs((width - step_size) - snake[0][0])

    # SEE NOTES ON WALL DISTANCE ALGORITHM
    for key in distWall:
        if key == "NE":
            if distWall["N"] > distWall["E"]:
                distWall["NE"] = 2 * distWall["E"]
            else:
                distWall["NE"] = 2 * distWall["N"]
        if key == "SE":
            if distWall["S"] > distWall["E"]:
                distWall["SE"] = 2 * distWall["E"]
            else:
                distWall["SE"] = 2 * distWall["S"]
        if key == "SW":
            if distWall["S"] > distWall["W"]:
                distWall["SW"] = 2 * distWall["W"]
            else:
                distWall["SW"] = 2 * distWall["S"]
        if key == "NW":
            if distWall["N"] > distWall["W"]:
                distWall["NW"] = 2 * distWall["W"]
            else:
                distWall["NW"] = 2 * distWall["N"]

    # APPLE BINARY
    hitApple = hitWall = False
    for direct in dirDict:
        laserX = snake[0][0]
        laserY = snake[0][1]
        hitWall = False
        while not (hitApple or hitWall):
            # HIT APPLE = succeed, break out of loop
            if laserX == apple[0] and laserY == apple[1]:
                dirDict[direct][2] = 1
                hitApple = True  # does it exit here, or update laser first?
            # HIT WALL = skip to next dir
            if laserX > width - step_size or laserX < 0:
                hitWall = True
            if laserY > height - step_size or laserY < 0:
                hitWall = True
            # has to be subtract for y, or swap dict above
            laserX += dirDict[direct][0] * step_size
            laserY -= dirDict[direct][1] * step_size
        if hitApple == True:
            break

    # SNAKE BODY BINARY
    hitSnake = hitWall = False
    for direct in dirDict:
        laserX = snake[0][0]
        laserY = snake[0][1]
        hitSnake = hitWall = False
        while not (hitSnake or hitWall):
            # HIT SNAKE = set to 1, continue searching directions
            for part in range(1, len(snake)):
                if laserX == snake[part][0] and laserY == snake[part][1]:
                    hitSnake = True
                    dirDict[direct][3] = 1
            # HIT WALL = skip to next dir
            if laserX > width - step_size or laserX < 0:
                hitWall = True
            if laserY > height - step_size or laserY < 0:
                hitWall = True
            # has to be subtract for y, or swap dict above
            laserX += dirDict[direct][0] * step_size
            laserY -= dirDict[direct][1] * step_size

    return distWall, dirDict


def refreshScreen(screen, width, step_size):
    screen.fill(white)
    for x in range(0, width, step_size):
        for y in range(0, height, step_size):
            gridRect = pygame.Rect(x, y, step_size, step_size)
            pygame.draw.rect(screen, black, gridRect, 1)


def eventHandle(running, prev_event, move):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # STORE VALUE OF KEY SO WE CANT MOVE BACKWARDS
        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_LEFT:
                # clean this to remove pass
                if prev_event == pygame.K_RIGHT:
                    pass
                else:
                    move = 'left'
                    prev_event = pygame.K_LEFT
                break


            elif event.key == pygame.K_RIGHT:
                if prev_event == pygame.K_LEFT:
                    pass
                else:
                    move = 'right'
                    prev_event = pygame.K_RIGHT
                break

            elif event.key == pygame.K_UP:
                if prev_event == pygame.K_DOWN:
                    pass
                else:
                    move = 'up'
                    prev_event = pygame.K_UP
                break

            elif event.key == pygame.K_DOWN:
                if prev_event == pygame.K_UP:
                    pass
                else:
                    move = 'down'
                    prev_event = pygame.K_DOWN
                break
                    
    return running, prev_event, move


def moveSnake(move, snake, previous):
    # MOVE HEAD
    if move == 'left':
        snake[0][0] -= step_size
    if move == 'right':
        snake[0][0] += step_size
    if move == 'up':
        snake[0][1] -= step_size
    if move == 'down':
        snake[0][1] += step_size

    # MOVE REST OF SNAKE
    if len(snake) > 1:
        for x in range(1, len(snake)):
            snake[x][0] = previous[x - 1][0]
            snake[x][1] = previous[x - 1][1]
            
    return snake

def checkCollisions(collide, snake, start_x, start_y, move, prev_event, score, width, height):
    collide = False
    running = True

    # WALL COLLISION
    if snake[0][0] <= -1:
        collide = True
    if (snake[0][0] + step_size) >= width + 1:
        collide = True
    if snake[0][1] <= -1:
        collide = True
    if (snake[0][1] + step_size) >= height + 1:
        collide = True

    # SNAKE COLLISION
    for i in range(1, len(snake)):
        if snake[i][0] == snake[0][0]:
            if snake[i][1] == snake[0][1]:
                collide = True
                break

    # RESET SNAKE - add game finish code here
    if collide == True:
        # remove all pieces except the head
        for i in range(len(snake) - 1, 0, -1):
            snake.remove(snake[i])
        snake[0][0] = start_x
        snake[0][1] = start_y
        collide = False
        move = 'right'
        prev_event = pygame.K_RIGHT
        score = 0
        running = False

    return collide, snake, move, prev_event, score, running

def checkApple(apple_here, snake, score, apple, previous):

    # DRAW GRID (SNAKE MOVESET)
    grid = []
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            grid.append([x, y])

    # IF APPLE EATEN
    if apple_here == False:
        apple_here = True

        for i in snake:
            if i in grid:
                grid.remove(i)
        apple = random.choice(grid)

    # DRAW APPLE
    if apple_here == True:
        appleRect = pygame.Rect(apple[0], apple[1], step_size, step_size)
        pygame.draw.rect(screen, green, appleRect)

    # EAT APPLE + GROW SNAKE - append new tail to prev pos of last tail element
    if snake[0][0] == apple[0] and snake[0][1] == apple[1]:
        apple_here = False
        score += 1
        snake.append(previous[-1])

    return apple_here, snake, score, apple


def drawSnake(snake, step_size):
    for i in range(len(snake)):
        colour = blue
        if i == 0:
            colour = red
        pygame.draw.rect(screen, colour, pygame.Rect(snake[i][0], snake[i][1], step_size, step_size))
        pygame.draw.rect(screen, red, pygame.Rect(snake[i][0], snake[i][1], step_size, step_size), 1)


def drawUI(width, height, menu_height, screen, distWall, dirDict, score):
    pygame.draw.rect(screen, grey, pygame.Rect(0, height, width, menu_height))

    # distText = font1.render(str(distWall), True, black)
    # screen.blit(distText, (0, 650))
    # dirText = font1.render(str(dirDict), True, black)
    # screen.blit(dirText, (0, 680))
    # scoreText = font2.render(str(score), True, purple)
    # screen.blit(scoreText, (540, 540))

    drawNN()

    # DRAW VISION (DEBUG MODE)
    x = 0
    for dir in dirDict:
        if dirDict[dir][2] == 1:
            text2 = font2.render(dir, True, purple)
            screen.blit(text2, (300,300))
        if dirDict[dir][3] == 1:
            text2 = font2.render(dir, True, orange)
            screen.blit(text2, (x,100))
        x += 60

def drawNN():
    struct = [10, 10, 10, 4]
    struct = [5, 4, 2]
    xwidth = 100
    ywidth = 50
    xshift = 50
    yshift = 50
    y_dyn = 0
    draw_lines = True
    nn = []

    # CONSTRUCT NETWORK = ABSOLUTE POSITIONS
    for layer in range(len(struct)):
        nn.append([])
        x = (layer * xwidth) + xshift
        for node in range(struct[layer]):
            if layer == 0:
                if struct[layer] >= struct[layer + 1]:
                    y_dyn = yshift
                else:
                    y_dyn = (math.floor((struct[layer + 1] - struct[layer]) / 2) * yshift) + yshift
            elif layer == len(struct) - 1:
                y_dyn = (math.floor((struct[layer - 1] - struct[layer]) / 2) * yshift) + yshift
            elif layer > 0 and layer < len(struct) - 1:
                if struct[layer] >= struct[layer - 1]:
                    y_dyn = yshift
                else:
                    y_dyn = (math.floor((struct[layer - 1] - struct[layer]) / 2) * yshift) + yshift
            y = (node * ywidth) + y_dyn
            nn[layer].append((x, y))

    # CREATE WEIGHTS for struct = [10, 10, 10, 4]
    # column vector for each node...
    # w1 = 2 * np.random.random((struct[1], struct[0])) - 1
    # w2 = 2 * np.random.random((struct[2], struct[1])) - 1
    # w3 = 2 * np.random.random((struct[3], struct[2])) - 1
    # w_all = [w1, w2, w3]

    # DRAW CIRCLES + LINES + APPLY WEIGHTS (COLOURS)
    for layer in range(len(nn)):
        for node in range(len(nn[layer])):
            x = nn[layer][node][0]
            y = nn[layer][node][1]
            if layer == len(nn) - 1:
                draw_lines = False

            if draw_lines:
                for next_node in range(len(nn[layer + 1])):
                    x2 = nn[layer + 1][next_node][0]
                    y2 = nn[layer + 1][next_node][1]

                    # weight = w_all[layer][next_node][node]

                    # lower weight = more blue = less activated
                    # between 1 and -1 is good, gives us the 2 normalized ranges
                    # if weight >= 0:
                    #     colour = [0, weight * 255, 0]
                    # else:
                    #     colour = [0, 0, -weight * 255]
                    colour = black
                    pygame.draw.line(screen, colour, (x + 600, y), (x2 + 600, y2), 2)

            # could just pass it the tuple here, instead of seperate x y
            pygame.draw.circle(screen, red, (x + 600, y), 10)

if __name__ == '__main__':
    # PYGAME VARIABLES
    pygame.init()
    logo = pygame.image.load("logo1.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("PROJECT APPLE")

    # FRAME VARIABLES
    width = 600
    height = 600
    step_size = 60  # Assuming a square snake
    menu_height = 100
    screen = pygame.display.set_mode((width + 600, height))
    font1 = pygame.font.Font('freesansbold.ttf', 10)
    font2 = pygame.font.Font('freesansbold.ttf', 60)

    # EVENT VARIABLES
    move = 'right'
    prev_event = pygame.K_LEFT

    # SNAKE / APPLE VARIABLES
    start_x = start_y = 60
    snake = [[start_x, start_y]]
    previous = [[]]
    score = 0
    distWall = {"N": 0, "NE": 0, "E": 0, "SE": 0, "S": 0, "SW": 0, "W": 0, "NW": 0}
    apple_here = False
    apple = 0
    collide = False

    # GAME LOOP
    running = True
    while (running):
        dirDict = {"N": [0, 1, 0, 0], "NE": [1, 1, 0, 0], "E": [1, 0, 0, 0], "SE": [1, -1, 0, 0], "S": [0, -1, 0, 0],
                   "SW": [-1, -1, 0, 0], "W": [-1, 0, 0, 0], "NW": [-1, 1, 0, 0]}
        refreshScreen(screen, width, step_size)
        running, prev_event, move = eventHandle(running, prev_event, move)
        previous = copy.deepcopy(snake)
        snake = moveSnake(move, snake, previous)
        collide, snake, move, prev_event, score, running = checkCollisions(collide, snake, start_x, start_y, move, prev_event, score, width, height)
        apple_here, snake, score, apple = checkApple(apple_here, snake, score, apple, previous)
        distWall, dirDict = updateVision(distWall, snake, step_size, dirDict, width, height, apple)

        if (len(snake) > 1):
            # TAIL DIRECTION
            tail_x = snake[-2][0] - snake[-1][0]
            tail_y = snake[-2][1] - snake[-1][1]

            # N S W E, or UP DN LFT RT
            tail_direction = [0, 0, 0, 0]
            if tail_y > 0:
                tail_direction[0] = 1
            if tail_y < 0:
                tail_direction[1] = 1
            if tail_x > 0:
                tail_direction[2] = 1
            if tail_x < 0:
                tail_direction[3] = 1
            print(tail_direction)



        drawSnake(snake, step_size)
        drawUI(width, height, menu_height, screen, distWall, dirDict, score)

        clk = pygame.time.Clock()
        clk.tick(4)
        pygame.display.flip()

#
# def mse(predict_output, train_output):
#     error = []
#     for i in range(len(train_output)):
#         for k in range(len(train_output[i])):
#             mse = (train_output[i][k] - predict_output[i][k]) ** 2
#             error.append(mse)
#     error = np.mean(np.array(error))
#     return error
#
# def activ_fn(x):
#     return 1 / (1 + np.exp(-x))
#
# # returns a list, each item is a weight matrix (np array)
# def create_wmatrix(struct):
#     weights = []
#     for i in range(len(struct) - 1):
#         # create the weights according to NN struct (no of nodes!)
#         # here the random weights can sometimes be around the same... use diff function?
#         # (row, col) form used, if doing opposite then apply .T operation elsewhere
#         rows_cols = (struct[i], struct[i + 1])
#         rand_weights = 2 * np.random.random(rows_cols) - 1
#         # rand_weights = rand_weights.T
#         weights.append(rand_weights)
#     return weights
#
#
# def runNNLoop(inputs, struct, weights):
#     # inputs = total array of input vectors
#     outputs = []
#     for i in range(len(inputs)):
#         single_output = runNN(inputs[i], struct, weights)
#         outputs.append(single_output)
#     return outputs
#
#
# def runNN(inputs, struct, weights):
#     # inputs is a single row vector
#     # (saved as a row in csv)
#
#     # LOOP HIDDEN LAYERS + OUTPUT
#     # matrix multiplication happens here!
#     outputs = None
#     for i in range(len(struct) - 1):
#         outputs = activ_fn(np.dot(inputs, weights[i]))
#         inputs = outputs
#     return outputs
#
#
# def populate(size, struct):
#     # each element in list contains the NN weights for each snake individual
#     # subelements = hidden layer weight matrices, built up using correct dimensions
#     # this form can be flattened to a single chromosome (used for crossover)
#     population = [None] * size
#     for i in range(len(population)):
#         individual_weights = create_wmatrix(struct)
#         population[i] = individual_weights
#     return population
#
#
# # mate 2 individuals - cross the weights together
# # p1 = entire weight matrix for one individual
# def mate(population, pop_size):
#     pop_new = []
#
#     # pair 2 individuals, so loop popsize/2 times
#     for _ in range(int(pop_size / 2)):
#
#         # pick random snakes (not based on fittest!)
#         # chance to pick the same 2
#         p1 = population[np.random.randint(0, len(population))]
#         p2 = population[np.random.randint(0, len(population))]
#         p1_flat = []
#         p2_flat = []
#
#         # flatten into 1D chromosome
#         for w_matrix in p1:
#             flat = w_matrix.flatten()
#             p1_flat = np.concatenate((p1_flat, flat))
#
#         for w_matrix in p2:
#             flat = w_matrix.flatten()
#             p2_flat = np.concatenate((p2_flat, flat))
#
#         # single point crossover using half-length
#         half_length = math.ceil(len(p1_flat) / 2)
#         child1 = np.concatenate(([p1_flat[:half_length], p2_flat[half_length:]]))
#         child2 = np.concatenate(([p2_flat[:half_length], p1_flat[half_length:]]))
#
#         # this shows that crossover has no effect really!
#         # child1 = copy.deepcopy(p1_flat)
#         # child2 = copy.deepcopy(p2_flat)
#
#         # mutate every single gene/weight
#         mutation1 = 2 * np.random.random((len(p1_flat),)) - 1
#         mutation2 = 2 * np.random.random((len(p1_flat),)) - 1
#         child1 = child1 + mutation1
#         child2 = child2 + mutation2
#
#         # wrap back into my form
#         p1_new = []
#         p2_new = []
#         index1 = 0
#         for w_matrix in p1:
#             rows, cols = w_matrix.shape
#             index2 = index1 + (rows * cols)
#             p1_new.append(np.reshape(np.array(child1[index1:index2]), (rows, cols)))
#             p2_new.append(np.reshape(np.array(child2[index1:index2]), (rows, cols)))
#             index1 = index2
#
#         pop_new.append(p1_new)
#         pop_new.append(p2_new)
#
#     return pop_new
#
#
# def genLoop(population, target, size, generations, struct, train_output):
#     # INITIAL POPULATION WITH RANDOM WEIGHTS
#     population = populate(size, struct)
#
#     for gen in range(generations):
#         gen += 1
#
#         # calc fitness for entire pop. select only top 10%
#         population, top_fitness = popLoop(population, target, size, struct, train_output)
#
#         print('Generation:', gen)
#         x = float(np.round(top_fitness, 20))
#         print('fittest: ', x)
#
#         # mate the top 10% to produce original size population
#         population = mate(population, size)
#
#         # Termination = best fitness = zero error!
#         if x <= 0:
#             break
#
#
# def popLoop(population, target, pop_size, struct, train_output):
#     # numpy slicing modifies orig array, so we make a duplicate to hold the sorted data!
#     population_sorted = []
#     fitness_list = []
#     individual_list = []
#     index_list = []
#     cord = []
#
#     for index, individual in enumerate(population):
#         # run the gameLoop for a single snake
#         predict_output = gameLoop(individual, struct)
#         error = mse(predict_output, train_output)
#         fitness = error
#
#         fitness_list.append(fitness)
#         individual_list.append(individual)
#         index_list.append(index)
#         cord.append((fitness, fitness))
#
#     # sort by best fitness - grab the indexes and put back into original population
#     sorted_fitness_list = np.sort(fitness_list)
#     sorted_fitness_index = np.array(fitness_list).argsort()
#     top_fitness = sorted_fitness_list[0]
#
#
#     for index in sorted_fitness_index:
#         population_sorted.append(population[index])
#
#     # return top 10% individuals + top fitness
#     top10 = math.ceil(pop_size * 0.1)
#     population_sorted = population_sorted[0:top10]
#
#     return population_sorted, top_fitness
#
#
# def gameLoop(individual, struct):
#
#
#
#
#
#
#     predict_output = runNNLoop(train_input, struct, individual)
#     return predict_output
#
#
# def draw(cord, target):
#     plot.xlim((-1, 1))
#     plot.ylim((-1, 1))
#     plot.scatter(cord[:, 0], cord[:, 1], c='green', s=12)
#     plot.scatter(target[0], target[1], c='red', s=60)
#
#
# # NEED TO SPLIT DATA HERE WITH TRAINING AND VALIDATION....
# # data = pd.read_csv(r'Data.csv')
# data = pd.read_csv(r'Data2.csv')
# data = data.sample(frac=1).reset_index(drop=True)
#
# # matrix will transpose it and also use [[values]] whereas array has [values]
# train_input_data = data.drop(['O1', 'O2'], axis = 1)
# train_input = np.array(train_input_data.values)
#
# train_output_data = data.drop(['A','B','C','D'], axis = 1)
# train_output = np.array(train_output_data.values)
# # print(train_output)
# #actual_outs = train_output
#
#
# # struct has to match the input and output data
# # very simple dataset so only one hidden layer works well
# size = 800
# # struct = [3, 10, 10, 4]
# struct = [4, 8, 2]
# # struct = [4, 8, 8, 8, 8, 4, 1]
# #inputs = [1, 2, 3]
# #w = create_wmatrix(struct)
#
# #outputs = runNN(inputs, struct, w)
# #print('Initial Guess:', outputs)
#
# target = 0
# generations = 100
# population = []
#
# #population = populate(size, struct)
# #pprint.pprint(population)
#
# # genloop = cycle
# # population, cord, answers = genLoop(population, target, size, generations, struct)
# genLoop(population, target, size, generations, struct, train_output)
# # draw(cord,(target,target))
# # for i in range(len(answers)):
# #     print(np.round(answers[i],2),'===',train_output[i])
#
#

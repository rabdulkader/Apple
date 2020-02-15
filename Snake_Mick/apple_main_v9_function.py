import pygame, time, random, copy
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plot
from scipy.special import expit as activ_fn

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
                if prev_event == pygame.K_RIGHT:
                    pass
                else:
                    move = 'left'
                    prev_event = pygame.K_LEFT

            elif event.key == pygame.K_RIGHT:
                if prev_event == pygame.K_LEFT:
                    pass
                else:
                    move = 'right'
                    prev_event = pygame.K_RIGHT

            elif event.key == pygame.K_UP:
                if prev_event == pygame.K_DOWN:
                    pass
                else:
                    move = 'up'
                    prev_event = pygame.K_UP

            elif event.key == pygame.K_DOWN:
                if prev_event == pygame.K_UP:
                    pass
                else:
                    move = 'down'
                    prev_event = pygame.K_DOWN
                    
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

    return collide, snake, move, prev_event, score

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


def drawUI(width, height, menu_height, screen, distWall, dirDict):
    pygame.draw.rect(screen, grey, pygame.Rect(0, height, width, menu_height))

    distText = font1.render(str(distWall), True, black)
    screen.blit(distText, (0, 650))
    dirText = font1.render(str(dirDict), True, black)
    screen.blit(dirText, (0, 680))

    x = 0
    for dir in dirDict:
        if dirDict[dir][2] == 1:
            text2 = font2.render(dir, True, purple)
            screen.blit(text2, (300,300))
        if dirDict[dir][3] == 1:
            text2 = font2.render(dir, True, orange)
            screen.blit(text2, (x,100))
        x += 60


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
    screen = pygame.display.set_mode((width, height + menu_height))
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
        collide, snake, move, prev_event, score = checkCollisions(collide, snake, start_x, start_y, move, prev_event, score, width, height)
        apple_here, snake, score, apple = checkApple(apple_here, snake, score, apple, previous)
        distWall, dirDict = updateVision(distWall, snake, step_size, dirDict, width, height, apple)
        drawSnake(snake, step_size)
        drawUI(width, height, menu_height, screen, distWall, dirDict)

        clk = pygame.time.Clock()
        clk.tick(5)
        pygame.display.flip()


import time
from collections import deque
import numpy as np
import pygame, sys, random
import torch
from pygame.math import Vector2
from ai import Linear_QNet, QTrainer

cell_size = 20
cell_number = 20


class FRUIT:
    def __init__(self, x=random.randint(0, cell_number - 1), y=random.randint(0, cell_number - 1)):
        self.pos = Vector2(x, y)

    def create(self):  # set pixel to fruit
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        pygame.draw.rect(screen, pygame.Color('red'), fruit_rect)


class SNAKE:
    def __init__(self):
        self.body = [Vector2(5, 10), Vector2(6, 10), Vector2(7, 10)]
        self.direction = Vector2(1, 0)

    def create(self):  # set pixels for snake body
        for body in self.body:
            body_rect = pygame.Rect(int(body.x * cell_size), int(body.y * cell_size), cell_size, cell_size)
            pygame.draw.rect(screen, pygame.Color('green'), body_rect)

    def move(self):  # update the snake's position
        self.body.append(self.body[-1] + self.direction)
        self.body.pop(0)


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class AGENT:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self):
        UP = snake.direction == Vector2(0, -1)
        DOWN = snake.direction == Vector2(0, 1)
        RIGHT = snake.direction == Vector2(1, 0)
        LEFT = snake.direction == Vector2(-1, 0)

        PIXEL_UP = snake.body[-1] + Vector2(0, -1)
        PIXEL_DOWN = snake.body[-1] + Vector2(0, 1)
        PIXEL_RIGHT = snake.body[-1] + Vector2(1, 0)
        PIXEL_LEFT = snake.body[-1] + Vector2(-1, 0)

        state = [
            # front danger
            (UP and (PIXEL_UP.y < 0 or PIXEL_UP in snake.body)) or
            (DOWN and (PIXEL_DOWN.y >= cell_number or PIXEL_DOWN in snake.body)) or
            (RIGHT and (PIXEL_RIGHT.x >= cell_number or PIXEL_RIGHT in snake.body)) or
            (LEFT and (PIXEL_LEFT.x < 0 or PIXEL_LEFT in snake.body)),

            # right danger
            (UP and (PIXEL_RIGHT.x >= cell_number or PIXEL_RIGHT in snake.body)) or
            (DOWN and (PIXEL_LEFT.x < 0 or PIXEL_LEFT in snake.body)) or
            (RIGHT and (PIXEL_DOWN.y >= cell_number or PIXEL_DOWN in snake.body)) or
            (LEFT and (PIXEL_UP.y < 0 or PIXEL_UP in snake.body)),

            # left danger
            (UP and (PIXEL_LEFT.x < 0 or PIXEL_LEFT in snake.body)) or
            (DOWN and (PIXEL_RIGHT.x >= cell_number or PIXEL_RIGHT in snake.body)) or
            (RIGHT and (PIXEL_UP.y < 0 or PIXEL_UP in snake.body)) or
            (LEFT and (PIXEL_DOWN.y >= cell_number or PIXEL_DOWN in snake.body)),

            # move direction
            UP,
            DOWN,
            LEFT,
            RIGHT,

            # fruit location
            fruit.pos.y < snake.body[-1].y,
            fruit.pos.y > snake.body[-1].y,
            fruit.pos.x < snake.body[-1].x,
            fruit.pos.x > snake.body[-1].x
            ]

        return(np.array(state, dtype = int))

    def get_move(self, state):
        self.epsilon = 80 - self.n_games  # 0 # exploration vs exploitation
        move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:  # random move (exploration)
            idx = random.randint(0, 2)
            move[idx] = 1
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))  # get move from model (exploitation)
            idx = torch.argmax(prediction).item()
            move[idx] = 1
        return(move)

    def train_short_memory(self, old_state, move, reward, new_state, done):  # train on action during game
        self.trainer.train_step(old_state, move, reward, new_state, done)

    def remember(self, old_state, move, reward, new_state, done):  # append action to long memory
        self.memory.append((old_state, move, reward, new_state, done))

    def train_long_memory(self):  # take a random sample from memory and train
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        old_state, move, reward, new_state, dones = zip(*mini_sample)
        self.trainer.train_step(old_state, move, reward, new_state, dones)


pygame.init()
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
clock = pygame.time.Clock()

fruit = FRUIT()
snake = SNAKE()
agent = AGENT()

SPEED = 40  # adjust game speed, higher == faster
reward = 0
frame = 0
move = [0, 0, 0]
over = False
record = 0
game_font = pygame.font.Font(None, 40)


def game_over():
    global over
    global record
    agent.n_games += 1
    over = True
    if len(snake.body) - 3 > record:  # update model with best snake
        record = len(snake.body) - 3
        agent.model.save()
        print('Model updated,', 'Record: ' + str(record))

    snake.body = [Vector2(5, 10), Vector2(6, 10), Vector2(7, 10)]  # reset snake, direction, and fruit
    snake.direction = Vector2(1, 0)
    fruit.pos = Vector2(random.randint(0, cell_number - 1), random.randint(0, cell_number - 1))

    agent.train_long_memory()  # train snake on long memory


def show_score():
    score_val = len(snake.body) - 3  # update game score
    score = game_font.render(str(score_val), True, pygame.Color('white'))
    screen.blit(score, (10, 10))


while True:
    reward = 0

    old_state = agent.get_state()  # 1. get current state of game

    move = agent.get_move(old_state)  # 2. get move based on forward pass of NN on state or random move on epsilon

    if move == [1, 0, 0]:                             # comment out for human game
        pass
    elif move == [0, 1, 0]:
        if snake.direction == Vector2(0, -1):
            snake.direction = Vector2(1, 0)
        elif snake.direction == Vector2(0, 1):
            snake.direction = Vector2(-1, 0)
        elif snake.direction == Vector2(1, 0):
            snake.direction = Vector2(0, 1)
        elif snake.direction == Vector2(-1, 0):
            snake.direction = Vector2(0, -1)
    elif move == [0, 0, 1]:
        if snake.direction == Vector2(0, -1):
            snake.direction = Vector2(-1, 0)
        elif snake.direction == Vector2(0, 1):
            snake.direction = Vector2(1, 0)
        elif snake.direction == Vector2(1, 0):
            snake.direction = Vector2(0, -1)
        elif snake.direction == Vector2(-1, 0):
            snake.direction = Vector2(0, 1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # if event.type == pygame.KEYDOWN:                                              # for human game
        #     if event.key == pygame.K_UP and snake.direction != Vector2(0, 1):
        #         snake.direction = Vector2(0, -1)
        #     elif event.key == pygame.K_DOWN and snake.direction != Vector2(0, -1):
        #         snake.direction = Vector2(0, 1)
        #     elif event.key == pygame.K_RIGHT and snake.direction != Vector2(-1, 0):
        #         snake.direction = Vector2(1, 0)
        #     elif event.key == pygame.K_LEFT and snake.direction != Vector2(1, 0):
        #         snake.direction = Vector2(-1, 0)

    screen.fill(pygame.Color('black'))

    snake.move()  # 3. move the snake

    if snake.body[-1] == fruit.pos:  # snake eat
        fruit.pos = Vector2(random.randint(0, cell_number - 1), random.randint(0, cell_number - 1))
        while fruit.pos in snake.body:
            fruit.pos = Vector2(random.randint(0, cell_number - 1), random.randint(0, cell_number - 1))
        reward += 10
        frame = 0
        snake.body.insert(0, snake.body[0])

    if snake.body[-1] in snake.body[:-1]:  # snake hits itself
        reward -= 10
        frame = 0
        game_over()

    if snake.body[-1].x < 0 or snake.body[-1].x >= cell_number \
            or snake.body[-1].y < 0 or snake.body[-1].y >= cell_number:  # snake out of bounds
        reward -= 10
        frame = 0
        game_over()

    if frame > 200:  # snake took too long
        reward -= 10
        frame = 0
        game_over()

    reward -= 0.1  # punish idling

    new_state = agent.get_state()  # 4. get new state after move, including reward or game over if occurs

    agent.train_short_memory(old_state, move, reward, new_state, over)  # 5. train data on NN

    agent.remember(old_state, move, reward, new_state, over)  # 6. save sample to train in long memory each game over

    fruit.create()
    snake.create()
    show_score()

    pygame.display.update()
    frame += 1
    # time.sleep(0.1)  # time delay for human game
    clock.tick(SPEED)  # adjust speed, higher faster

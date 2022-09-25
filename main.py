import pygame, sys, random
from pygame.math import Vector2

cell_size = 20
cell_number = 20

class FRUIT:
    def __init__(self, x = random.randint(0, cell_number - 1), y = random.randint(0, cell_number - 1)):
        self.pos = Vector2(x, y)

    def create(self):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size), int(self.pos.y * cell_size), cell_size, cell_size)
        pygame.draw.rect(screen, pygame.Color('red'), fruit_rect)

class SNAKE:
    def __init__(self):
        self.body = [Vector2(5, 10), Vector2(6, 10), Vector2(7, 10)]
        self.direction = Vector2(1, 0)

    def create(self):
        for body in self.body:
            body_rect = pygame.Rect(int(body.x * cell_size), int(body.y * cell_size), cell_size, cell_size)
            pygame.draw.rect(screen, pygame.Color('green'), body_rect)

    def move(self):
        self.body.append(self.body[-1] + self.direction)
        self.body.pop(0)

pygame.init()
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
clock = pygame.time.Clock()

fruit = FRUIT()
snake = SNAKE()

def game_over():
    pygame.quit()
    sys.exit()

game_font = pygame.font.Font(None, 40)
def show_score():
    score_val = len(snake.body) - 3
    score = game_font.render(str(score_val), True, pygame.Color('white'))
    screen.blit(score, (10, 10))

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 150)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over()
        if event.type == SCREEN_UPDATE:
            snake.move()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and snake.direction != Vector2(0, 1):
                snake.direction = Vector2(0, -1)
            if event.key == pygame.K_DOWN and snake.direction != Vector2(0, -1):
                snake.direction = Vector2(0, 1)
            if event.key == pygame.K_RIGHT and snake.direction != Vector2(-1, 0):
                snake.direction = Vector2(1, 0)
            if event.key == pygame.K_LEFT and snake.direction != Vector2(1, 0):
                snake.direction = Vector2(-1, 0)
    screen.fill(pygame.Color('black'))

    if snake.body[-1] == fruit.pos: # snake eat
        fruit = FRUIT(random.randint(0, cell_number - 1), random.randint(0, cell_number - 1))
        while fruit.pos in snake.body:
            fruit = FRUIT(random.randint(0, cell_number - 1), random.randint(0, cell_number - 1))
        snake.body.insert(0, snake.body[0])

    if snake.body[-1] in snake.body[:-1]: # snake hits itself
        game_over()

    if snake.body[-1].x < 0 or snake.body[-1].x >= cell_number \
       or snake.body[-1].y < 0 or snake.body[-1].y >= cell_number: # snake out of bounds
        game_over()

    fruit.create()
    snake.create()
    show_score()
    pygame.display.update()
    clock.tick(60)

import pygame
import time
import random

pygame.init()
WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RunJumpS")
clock = pygame.time.Clock()

def start_screen():
    WIN.fill("black")
    pygame.image.load("pygame_powered_lowres")

def main():
    run = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    
    
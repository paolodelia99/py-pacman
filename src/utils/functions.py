import pygame as pg


def get_image_surface(file_name: str):
    return pg.image.load('res/tiles/' + file_name).convert()

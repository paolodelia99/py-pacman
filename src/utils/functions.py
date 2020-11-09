import pygame as pg


def get_image_surface(file_path: str):
    return pg.image.load(file_path).convert()

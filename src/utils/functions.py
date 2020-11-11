import pygame as pg


def get_image_surface(file_path: str) -> pg.Surface:
    return pg.image.load(file_path).convert()

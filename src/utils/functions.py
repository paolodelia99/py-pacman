import pygame as pg


def get_image_surface(file_path: str) -> pg.Surface:
    return pg.image.load(file_path).convert()


def check_if_hit(x1: int, y1: int, x2: int, y2: int, threshold: int) -> bool:
    if (x1 - x2 < threshold) and (x1 - x2 > -threshold) \
            and (y1 - y2 < threshold) and (y1 - y2 > -threshold):
        return True
    else:
        return False

import pygame as pg
import pygame_menu

from pygame_menu.themes import Theme
from src.game import Game


class Controller(object):

    def __init__(self, layout_name:str):
        self.screen = pg.display.set_mode((600, 400))
        self.menu_theme = Theme(
            title_font=pygame_menu.font.FONT_8BIT,
            widget_font=pygame_menu.font.FONT_8BIT
        )
        self.layout_name = layout_name

    def load_level(self):
        game = Game(
            screen=self.screen,
            layout_name=self.layout_name
        )
        game.start_game()

    def load_menu(self):
        menu = pygame_menu.Menu(400, 600, 'Welcome',
                                theme=self.menu_theme)

        menu.add_button('Play', self.load_level())
        menu.add_button('Quit', pygame_menu.events.EXIT)

        menu.mainloop(self.screen)

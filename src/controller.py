import pygame as pg
import pygame_menu
from pygame_menu.themes import Theme


class Controller(object):

    def __init__(self):
        self.screen = pg.display.set_mode((600, 400))
        self.menu_theme = Theme(
            title_font=pygame_menu.font.FONT_8BIT,
            widget_font=pygame_menu.font.FONT_8BIT
        )

    def load_level(self):
        print('loading the level...')
        pass

    def load_menu(self):
        menu = pygame_menu.Menu(400, 600, 'Welcome',
                                theme=self.menu_theme)

        menu.add_button('Play', self.load_level())
        menu.add_button('Quit', pygame_menu.events.EXIT)

        menu.mainloop(self.screen)

import pygame as pg
import pygame_menu
from pygame_menu.themes import Theme

from src.game import Game
from .map import Map


class Controller(object):

    def __init__(self, layout_name: str, act_sound: bool, act_state: bool, **kwargs):
        pg.init()
        self.layout_name = layout_name
        self.act_sound = act_sound
        self.act_state = act_state
        self.ai_agent = kwargs['ai_agent']
        self.maze = Map(layout_name)
        self.width, self.height = self.maze.get_map_sizes()
        self.screen = Controller.get_screen(act_state, self.width, self.height)
        self.menu_theme = Theme(
            title_font=pygame_menu.font.FONT_8BIT,
            widget_font=pygame_menu.font.FONT_8BIT
        )
        pg.display.set_caption("Pacman")

    def load_level(self):
        game = Game(
            maze=self.maze,
            screen=self.screen,
            sounds_active=self.act_sound,
            state_active=self.act_state,
            agent=self.ai_agent
        )
        game.start_game()

    @staticmethod
    def get_screen(act_state: bool, width: int, height: int) -> pg.SurfaceType:
        if act_state:
            return pg.display.set_mode(((width * 2) + 48, height))
        else:
            return pg.display.set_mode((width, height))

    def load_menu(self):
        menu = pygame_menu.Menu(self.height, self.width,
                                'Welcome', theme=self.menu_theme)

        menu.add_button('Play', self.load_level)
        menu.add_button('Quit', pygame_menu.events.EXIT)

        menu.mainloop(self.screen)

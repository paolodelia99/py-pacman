import os
import sys

import pygame as pg
from pygame.mixer import SoundType
from pygame.surface import SurfaceType

from src.pacman import Pacman
from .constants import GHOST_COLORS
from .ghost import Ghost
from .map import Map
from .utils.functions import get_image_surface
from .utils.game_mode import GameMode
from .utils.path_finder import PathFinder


class Game(object):
    channel_background: pg.mixer.Channel
    clock: pg.time.Clock
    snd_intro: SoundType
    snd_default: SoundType
    snd_death: SoundType
    screen_bg: object
    num_digits: dict
    imGameOver: SurfaceType
    imReady: SurfaceType
    imLife: SurfaceType

    def __init__(self, maze: Map, screen, sounds_active: bool):
        self.screen = screen
        self.score = 0
        self.sounds_active = sounds_active
        self.maze = maze
        self.maze.build_tile_map()

        self.is_run = True
        self.is_game_run = False
        self.pause = False

        self.init_mixer()
        self.load_assets()
        self.fruitType = None

        self.player = Pacman(sounds_active=self.sounds_active)
        self.ghosts = [Ghost(i, GHOST_COLORS[i]) for i in range(0, 4)]
        self.path_finder = PathFinder()

        self.set_game_mode(1)  # fixme : just a stub for now

    def load_assets(self):
        self.screen_bg = get_image_surface(os.path.join('res', 'backgrounds', '1.gif'))
        self.num_digits = {
            i: get_image_surface(os.path.join(sys.path[0], "res", "text", str(i) + ".gif"))
            for i in range(0, 10)
        }
        self.imGameOver = get_image_surface(os.path.join(sys.path[0], "res", "text", "gameover.gif"))
        self.imReady = get_image_surface(os.path.join(sys.path[0], "res", "text", "ready.gif"))
        self.imLife = get_image_surface(os.path.join(sys.path[0], "res", "text", "life.gif"))
        if self.sounds_active:
            self.snd_intro = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "levelintro.wav"))
            self.snd_default = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "default.wav"))
            self.snd_death = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "death.wav"))

    def init_mixer(self):
        pg.mixer.init()
        pg.mixer.set_num_channels(7)
        self.channel_background = pg.mixer.Channel(6)

    def init_game(self):
        self.clock = pg.time.Clock()
        pg.mouse.set_visible(False)

    def play_bkg_sound(self, snd):
        self.channel_background.stop()
        self.channel_background.play(snd, loops=-1)

    def init_players_in_map(self):
        home_x, home_y = self.maze.get_player_home()
        self.player.init_position(home_x, home_y)

    def init_screen(self):
        self.screen.blit(self.screen_bg, (0, 0))

    def start_game(self):
        self.load_assets()
        self.game_loop()

    def game_loop(self):
        self.init_game()
        self.init_players_in_map()

        while self.is_run:
            self.init_screen()
            self.event_loop()
            self.draw()

            if self.game_mode in [GameMode.normal, GameMode.change_ghosts, GameMode.wait_after_eating_ghost]:
                self.player.move(self.maze, self)

            pg.display.flip()
            self.clock.tick(60)

    def event_loop(self):
        self.player.check_keyboard_inputs(self.maze)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.is_run = False
                    self.channel_background.stop()

    def draw(self):
        self.maze.draw(self.screen)
        self.player.draw(self.screen, self.game_mode)
        # fixme: draw score and lives

    def draw_texts(self):
        # draw_score
        # draw_ready
        # draw_game_over
        pass

    def set_game_mode(self, mode: int):
        self.game_mode = GameMode(mode)
        self.set_proper_bkg_music()

    def add_score(self, score_to_add: int):
        self.score += score_to_add

    def set_proper_bkg_music(self):
        if self.game_mode == GameMode.normal:
            self.play_bkg_sound(self.snd_default)

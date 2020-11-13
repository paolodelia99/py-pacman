import os
import sys
import threading
from typing import Union, Tuple

import pygame as pg
from pygame.mixer import SoundType
from pygame.surface import SurfaceType

from src.pacman import Pacman
from .constants import GHOST_COLORS, TILE_SIZE, SCORE_COLWIDTH, MODES_TO_ZERO
from .ghost import Ghost
from .map import Map
from .utils.functions import get_image_surface
from .utils.game_mode import GameMode
from .utils.ghost_state import GhostState
from .utils.path_finder import PathFinder


class Game(object):
    channel_background: pg.mixer.Channel
    clock: pg.time.Clock
    snd_intro: SoundType
    snd_default: SoundType
    snd_death: SoundType
    snd_extra_pac: SoundType
    screen_bg: object
    num_digits: dict
    img_game_over: SurfaceType
    img_ready: SurfaceType
    img_life: SurfaceType

    def __init__(self, maze: Map, screen, sounds_active: bool):
        self.screen = screen
        self.screen_size = {
            "height": pg.display.Info().current_h,
            "width": pg.display.Info().current_w
        }
        self.score = 0
        self.mode_timer = 0
        self.ghosts_timer = 0
        self.value_to_draw = 0
        self.sounds_active = sounds_active
        self.maze = maze
        self.maze.build_tile_map()

        self.is_run = True
        self.is_game_run = False
        self.pause = False
        self.draw_value = False

        if self.sounds_active:
            self.init_mixer()
        self.load_assets()

        self.player = Pacman(sounds_active=self.sounds_active)
        self.ghosts = [Ghost(i, GHOST_COLORS[i]) for i in range(0, 4)]
        self.path_finder = PathFinder()

    def load_assets(self):
        self.screen_bg = get_image_surface(os.path.join('res', 'backgrounds', '1.gif'))
        self.num_digits = {
            i: get_image_surface(os.path.join(sys.path[0], "res", "text", str(i) + ".gif"))
            for i in range(0, 10)
        }
        self.img_game_over = get_image_surface(os.path.join(sys.path[0], "res", "text", "gameover.gif"))
        self.img_ready = get_image_surface(os.path.join(sys.path[0], "res", "text", "ready.gif"))
        self.img_life = get_image_surface(os.path.join(sys.path[0], "res", "text", "life.gif"))
        if self.sounds_active:
            self.snd_intro = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "levelintro.wav"))
            self.snd_default = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "default.wav"))
            self.snd_death = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "death.wav"))
            self.snd_extra_pac = pg.mixer.Sound(os.path.join(sys.path[0], "res", "sounds", "extrapac.wav"))

    def init_mixer(self):
        pg.mixer.init()
        pg.mixer.set_num_channels(7)
        self.channel_background = pg.mixer.Channel(6)

    def init_game(self):
        self.clock = pg.time.Clock()
        pg.mouse.set_visible(False)

    def play_bkg_sound(self, snd, loops=-1):
        self.channel_background.stop()
        self.channel_background.play(snd, loops=loops)

    def init_players_in_map(self):
        home_x, home_y = self.maze.get_player_home()
        self.player.init_home(home_x, home_y)
        ghosts_home = self.maze.get_ghosts_home(len(self.ghosts))
        for i, ghost in enumerate(self.ghosts):
            ghost.init_home(ghosts_home[i]["x"], ghosts_home[i]["y"])

    def init_screen(self):
        self.screen.blit(self.screen_bg, (0, 0))

    def start_game(self, restart=False):
        if restart:
            self.maze.reinit_map()

        self.set_mode(0)
        self.init_game()
        self.init_players_in_map()
        self.game_loop()

    def game_loop(self):
        while self.is_run:
            self.init_screen()
            self.event_loop()
            self.draw()

            self.check_game_mode()

            # control pacman
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
                    self.start_game(restart=True)
                    if self.sounds_active:
                        self.channel_background.stop()

    def draw(self):
        th1 = threading.Thread(target=self.maze.draw, args=(self.screen,))
        th2 = threading.Thread(target=self.draw_texts)
        th3 = threading.Thread(target=self.player.draw, args=(self.screen, self.game_mode,))

        th1.start()
        th3.start()
        th2.start()

        for ghost in self.ghosts:
            ghost.draw(self.screen, self, self.player)

        th1.join()
        th2.join()
        th3.join()

    def draw_texts(self):
        self.draw_number(self.score, 0, self.maze.shape[0] * TILE_SIZE)

        for i in range(0, self.player.lives):
            self.screen.blit(
                self.img_life, (
                    (self.maze.shape[1] // 2 - 1) * TILE_SIZE + i * 10,
                    self.maze.shape[0] * TILE_SIZE))

        if self.game_mode == 3:
            self.screen.blit(self.img_game_over, self.set_text_center(self.img_game_over))
        elif self.game_mode in [GameMode.ready, GameMode.wait_to_start]:
            self.screen.blit(self.img_ready, self.set_text_center(self.img_ready))

    def set_text_center(self, img: SurfaceType) -> Tuple[int, int]:
        return self.screen_size["width"] // 2 - (img.get_width() // 2), \
               self.screen_size["height"] // 2 - (img.get_height() // 2)

    def set_mode(self, mode: Union[int, GameMode]):
        self.game_mode = GameMode(mode) if type(mode) is int else mode
        if self.game_mode in MODES_TO_ZERO:
            self.mode_timer = 0
        if self.sounds_active:
            self.set_proper_bkg_music()

    def add_score(self, score_to_add: int):
        self.score += score_to_add

    def set_proper_bkg_music(self):
        if self.game_mode == GameMode.ready:
            self.play_bkg_sound(self.snd_intro)
        elif self.game_mode == GameMode.normal:
            self.play_bkg_sound(self.snd_default)
        elif self.game_mode == GameMode.hit_ghost:
            self.play_bkg_sound(self.snd_death, 1)
        elif self.game_mode == GameMode.change_ghosts \
                or self.are_ghosts_vulnerable():
            self.play_bkg_sound(self.snd_extra_pac)
        else:
            self.channel_background.stop()

    def draw_number(self, num: int, x: int, y: int):
        str_num = str(num)

        for i in range(0, len(str_num)):
            digit = int(str_num[i])
            self.screen.blit(self.num_digits[digit], (x + i * SCORE_COLWIDTH, y))

    def check_game_mode(self):
        if self.game_mode == GameMode.ready:
            if self.mode_timer == 264:
                self.set_mode(1)
        elif self.game_mode == GameMode.normal:
            for ghost in self.ghosts:
                # fixme: make the ghost move ghost.move()
                pass
        elif self.game_mode == GameMode.hit_ghost:
            if self.mode_timer == 90:
                self.restart()
                self.player.lives -= 1
                if self.player.lives == -1:
                    self.set_mode(GameMode.game_over)
                else:
                    self.set_mode(GameMode.wait_to_start)
        elif self.game_mode == GameMode.game_over:
            if pg.key.get_pressed()[pg.K_RETURN]:
                self.start_game(restart=True)
        elif self.game_mode == GameMode.wait_to_start:
            if self.mode_timer == 60:
                self.set_mode(GameMode.normal)
                self.player.vel_x = self.player.speed
        elif self.game_mode == GameMode.wait_after_eating_ghost:
            if self.draw_value and self.mode_timer < 20:
                self.draw_number(self.value_to_draw, self.player.x - 10, self.player.y - 20)
            else:
                self.draw_value = False
                self.value_to_draw = 0
        elif self.game_mode == GameMode.wait_after_finishing_level:
            pass
        elif self.game_mode == GameMode.change_ghosts:
            pass

        self.check_ghosts_state()

        self.mode_timer += 1

    def check_ghosts_state(self):
        if self.are_ghosts_vulnerable():
            if self.ghosts_timer == 360:
                self.set_mode(GameMode.normal)
                for ghost in self.ghosts:
                    ghost.set_normal()
            else:
                self.ghosts_timer += 1

    def are_ghosts_vulnerable(self) -> bool:
        return [ghost.state == GhostState.vulnerable for ghost in self.ghosts].count(True) > 0

    def make_ghosts_vulnerable(self):
        self.ghosts_timer = 0
        for ghost in self.ghosts:
            ghost.set_vulnerable()

    def duplicate_vulnerable_ghosts_value(self):
        for ghost in self.ghosts:
            if ghost.state == GhostState.vulnerable:
                ghost.duplicate_value()

    def restart(self):
        self.init_players_in_map()
        self.player.set_start_anim()
        self.player.set_vel_to_zero()

    def draw_ghost_value(self, value):
        self.draw_value = True
        self.value_to_draw = value

import os
import pygame as pg

from src.utils.ghost_state import GhostState


class Ghost(object):

    def __init__(self, ghost_id, ghost_color):
        self.x = 0
        self.y = 0
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 1
        self.nearest_row = 0
        self.nearest_col = 0
        self.id = ghost_id
        self.state = GhostState.normal

        self.home_x = 0
        self.home_y = 0

        self.current_path = ""
        self.ghost_color = ghost_color

        self.anim = Ghost.load_ghost_animation(self.ghost_color)

        self.anim_fram = 1
        self.anim_delay = 0

    @staticmethod
    def load_ghost_animation(color):
        anim = {}

        for i in range(1, 7):
            anim[i] = pg.image.load(
                os.path.join("res", "sprite", "ghost " + str(i) + ".gif")).convert()

            # change the ghost color in this frame
            for y in range(0, 16):
                for x in range(0, 16):

                    if anim[i].get_at((x, y)) == (255, 0, 0, 255):
                        # default, red ghost body color
                        anim[i].set_at((x, y), color)

        return anim

    def draw(self, screen, game, player, level):

        if game.mode == 3:
            return False

        # ghost eyes --
        for y in range(4, 8, 1):
            for x in range(3, 7, 1):
                self.anim[self.anim_fram].set_at((x, y), (255, 255, 255, 255))
                self.anim[self.anim_fram].set_at((x + 6, y), (255, 255, 255, 255))

                if player.x > self.x and player.y > self.y:
                    # player is to lower-right
                    pupil_set = (5, 6)
                elif player.x < self.x and player.y > self.y:
                    # player is to lower-left
                    pupil_set = (3, 6)
                elif player.x > self.x and player.y < self.y:
                    # player is to upper-right
                    pupil_set = (5, 4)
                elif player.x < self.x and player.y < self.y:
                    # player is to upper-left
                    pupil_set = (3, 4)
                else:
                    pupil_set = (4, 6)

        for y in range(pupil_set[1], pupil_set[1] + 2, 1):
            for x in range(pupil_set[0], pupil_set[0] + 2, 1):
                self.anim[self.anim_fram].set_at((x, y), (0, 0, 255, 255))
                self.anim[self.anim_fram].set_at((x + 6, y), (0, 0, 255, 255))

        if self.state == GhostState.normal:
            # draw regular ghost (this one)
            screen.blit(self.anim[self.anim_fram],
                        (self.x - game.screen_pixel_pos[0], self.y - game.screen_pixel_pos[1]))
        elif self.state == GhostState.vulnerable:
            if game.ghost_timer > 100:
                # blue
                screen.blit(Ghost.load_ghost_animation((50, 50, 255, 255))[self.anim_fram],
                            (self.x - game.screen_pixel_pos[0], self.y - game.screen_pixel_pos[1]))
            else:
                # blue/white flashing
                temp_timer_i = int(game.ghost_timer / 10)
                if temp_timer_i == 1 or temp_timer_i == 3 or temp_timer_i == 5 or temp_timer_i == 7 or temp_timer_i == 9:
                    screen.blit(Ghost.load_ghost_animation((255, 255, 255, 255))[self.anim_fram],
                                (self.x - game.screen_pixel_pos[0], self.y - game.screen_pixel_pos[1]))
                else:
                    screen.blit(Ghost.load_ghost_animation((50, 50, 255, 255))[self.anim_fram],
                                (self.x - game.screen_pixel_pos[0], self.y - game.screen_pixel_pos[1]))

        elif self.state == GhostState.spectacles:
            # draw glasses
            screen.blit(level.tile_id_image[level.tile_id['glasses']],
                        (self.x - game.screen_pixel_pos[0], self.y - game.screen_pixel_pos[1]))

        if game.mode == 6 or game.mode == 7:
            # don't animate ghost if the level is complete
            return False

        self.anim_delay += 1

        if self.anim_delay == 2:
            self.anim_fram += 1

            if self.anim_fram == 7:
                # wrap to beginning
                self.anim_fram = 1

            self.anim_delay = 0


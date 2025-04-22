import numpy as np

class RewardManager:
    def __init__(self, generation):
        self.generation = generation
        self.prev_distance = None
        self.total_reward = 0
        self.apple_eaten = 0
        self.seen_squares = []
        self.distance = 0

    def reset_distance(self, head_pos, apple_pos):
        self.prev_distance = self._manhattan(head_pos, apple_pos)

    def update_distance(self, head_pos, apple_pos):
        explored = self.exploration_check(head_pos)

        if self.prev_distance is None:
            return 0
        else:
            new_dist = self._manhattan(head_pos, apple_pos)
            delta = self.prev_distance - new_dist
            self.distance = new_dist
            # self.total_reward += delta * 3 + 1
            self.prev_distance = new_dist
            return delta

    def exploration_check(self, pos):
        if pos in self.seen_squares:
            # self.total_reward -= min(0, 2 * self.seen_squares.count(pos))
            return False
        else:
            self.total_reward += 0
            self.seen_squares.append(pos)
            return True
        

    def exploration_reset(self):
        self.seen_squares = []
        return

    def ate_apple(self):
        self.apple_eaten += 1
        self.total_reward += 1
        self.exploration_reset()
        return self.apple_eaten

    def loop_penalty(self, repeated_tiles):
        self.total_reward -= repeated_tiles * 0
        return repeated_tiles

    def survival_bonus(self, steps):
        self.total_reward += 0.1

    def hunger_penalty(self, steps):
        self.total_reward -= 0 #min(10, max(0, 2 * (steps - 100)))

    def get_total(self):
        return max(-100, self.total_reward)
    
    def death_penalty(self):
        # base_penalty = -5
        # scaled_penalty = -1 * length * .1  # tweak these
        self.total_reward -= 10
        self.apple_eaten = 0 
        self.exploration_reset()

    def get_distance(self):
        return self.distance
    
    def line_reward(self, move_idx, last_move_idx):
        if move_idx != last_move_idx:
            self.total_reward += 0


    @staticmethod
    def _manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class RewardManager:
    def __init__(self, generation):
        self.generation = generation
        self.prev_distance = None
        self.total_reward = 0
        self.apple_eaten = 0

    def reset_distance(self, head_pos, apple_pos):
        self.prev_distance = self._manhattan(head_pos, apple_pos)

    def update_distance(self, head_pos, apple_pos):
        if self.prev_distance is None:
            return 0
        new_distance = self._manhattan(head_pos, apple_pos)
        delta = self.prev_distance - new_distance
        self.prev_distance = new_distance

        if delta > 0:
            self.total_reward += delta * 10
        elif delta < 0:
            self.total_reward += delta * 10 - self.apple_eaten * .01

        return delta

    def ate_apple(self):
        self.apple_eaten += 1
        self.total_reward += 100
        return self.apple_eaten

    def loop_penalty(self, repeated_tiles):
        self.total_reward -= repeated_tiles * 3
        return repeated_tiles

    def survival_bonus(self, steps):
        self.total_reward += steps * 0.01

    def get_total(self):
        return self.total_reward
    
    def death_penalty(self, steps, length):
        base_penalty = -10
        scaled_penalty = -1 * (steps * 0.01 + length * 2)  # tweak these
        self.total_reward += min(base_penalty, scaled_penalty)


    @staticmethod
    def _manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

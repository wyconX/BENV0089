import numpy as np


class FGPolicy:
    def __init__(self, ep_min=0.01, ep_decay=0.0001, esp_total=1000, ep_max=1):
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.eps_total = esp_total
        self.ep_max = ep_max

    def epsilon(self, step):
        if self.eps_total == 1:
            return self.ep_min
        return max(self.ep_min, self.ep_max - (self.ep_max - self.ep_min) * step / self.eps_total)

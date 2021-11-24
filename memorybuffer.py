import random 
random.seed()


class Memory:

    def __init__(self, size):

        self.size = size 
        self.step_list = []

    def __len__(self):

        return len(self.step_list)

    def add_step(self, step):

        self.step_list.append(step)

        if len(self.step_list) > self.size:
            self.step_list = self.step_list[len(self.step_list) - self.size:]

    def get_random_steps(self, fetch_size):

        return random.sample(self.step_list, fetch_size)
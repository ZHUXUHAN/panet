from bisect import bisect_right


class Adjust_Learning_Rate(object):
    def __init__(self, optimizer, solver_dict):

        self.iteration = 0
        self.optimizer = optimizer
        self.gamma = solver_dict['gamma']
        self.steps = solver_dict['steps']
        self.new_lr = solver_dict['base_lr']
        self.base_lr = solver_dict['base_lr']
        self.warm_up_iters = solver_dict['warm_up_iters']
        self.warm_up_factor = solver_dict['warm_up_factor']

    def get_lr(self):
        if self.iteration <= self.warm_up_iters:  # warm up
            alpha = self.iteration / self.warm_up_iters
            warmup_factor = self.warm_up_factor * (1 - alpha) + alpha
            new_lr = self.base_lr * warmup_factor
        else:
            new_lr = self.base_lr * self.gamma ** bisect_right(self.steps, self.iteration)

        return new_lr

    def update_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.new_lr

    def step(self, cur_iter=None):
        if cur_iter is None:
            cur_iter = self.iteration + 1
        self.iteration = cur_iter
        # update learning rate
        self.new_lr = self.get_lr()
        self.update_learning_rate()


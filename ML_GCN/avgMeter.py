class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.n = 0
        self.var = 0

    def add(self, val, n=1):
        self.sum += val * n
        self.n += n
        self.var += n*val*val

    def value(self):
        avg = self.sum /self.n
        std = (self.var - self.n *avg*avg)/(self.n - 1)
        std = std ** 0.5
        return [avg, std]

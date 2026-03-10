class EarlyStopper:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')

    def check(self, loss):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
            best = True
            stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                best = False
                stop = True
            else:
                best = False
                stop = False
        return best, stop

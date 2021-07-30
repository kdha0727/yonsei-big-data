class LearningReporthook(object):
    """Logging hook that leave logs in stdout."""

    # Constructor. initialized in each epoch.
    def __init__(self, epoch, train_iter, val_iter, stream=None):
        self.train_batch_size = train_iter.batch_size
        self.train_loader_length = len(train_iter)
        self.train_dataset_length = len(train_iter.dataset)
        self.val_dataset_length = len(val_iter.dataset)
        self.log(f'Epoch {epoch}', end='\n')
        if stream is not None:
            self.stream = stream

    # Logging method that is used in train function
    def train(self, loss, iteration, data=None):
        data_length = len(data) if data is not None else self.train_batch_size
        message = (
            f'\r[Train]\t '
            f'Progress: {iteration * data_length}/{self.train_dataset_length} '
            f'({100. * iteration / self.train_loader_length:.2f}%), '
            f'\tLoss: {loss.item():.6f}'
        )
        self.log(message, end=' ')

    # Logging method that is used in test function
    def evaluate(self, loss, correct):
        message = (
            f'\n[Eval]\t '
            f'Average loss: {loss:.5f}, '
            f'\t\tTotal accuracy: {100. * correct:.2f}%'
        )
        self.log(message, end='\n\n')

    log = staticmethod(print)  # In notebook, using built-in is better

# Initialize project - import frameworks and load dataset

#
# Configurations
#

# Model Configuration
BATCH_SIZE = 64
USE_CUDA = True

# File I/O Configuration
BASE_DIR = './'
DATASET_FOLDER = 'data'
SNAPSHOT_FOLDER = 'snapshot'


#
# Import modules and set environments
#

# Framework import
import os  # noqa
import torch  # noqa
import torch.nn as nn  # noqa
import torch.nn.functional as F  # noqa
import torch.optim  # noqa
import torchtext  # noqa
if os.name == "nt":
    from eunjeon import Mecab  # noqa
else:
    from konlpy.tag import Mecab  # noqa

# Utility import
import sys  # noqa
import os  # noqa
import platform  # noqa
import urllib.request  # noqa
import matplotlib.pyplot as plt  # noqa

# Reporthook import
try:
    import reporthook
except ImportError:
    reporthook = None

# Prepare cuda-related variables
USE_CUDA = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Environment check
print(
    "ENVIRONMENT INFORMATION\n"
    "OS version: \t\t{0}\n"
    "Python version:\t\t{1}\n"
    "Torch version:\t\t{2}\n"
    "Torch device:\t\t{3}\n".format(platform.platform(), sys.version, torch.__version__, device)
)


#
# Prepare dataset
#

# Set directories
BASE_DIR = os.path.abspath(BASE_DIR)
dataset_dir = os.path.join(BASE_DIR, DATASET_FOLDER)
snapshot_dir = os.path.join(BASE_DIR, SNAPSHOT_FOLDER)
os.makedirs(dataset_dir, exist_ok=True)

# Load tokenizer
tokenizer = Mecab()

# Set fields
ID = torchtext.legacy.data.Field(
    sequential=False,
    use_vocab=False)
TEXT = torchtext.legacy.data.Field(
    sequential=True,
    use_vocab=True,
    tokenize=tokenizer.morphs,
    lower=True,
    batch_first=True,
    fix_length=20)
LABEL = torchtext.legacy.data.Field(
    sequential=False,
    use_vocab=False,
    is_target=True)

# Download dataset
if not os.path.exists(os.path.join(dataset_dir, "ratings_train.txt")):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
        filename=os.path.join(dataset_dir, "ratings_train.txt"))
if not os.path.exists(os.path.join(dataset_dir, "ratings_test.txt")):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
        filename=os.path.join(dataset_dir, "ratings_test.txt"))

# Load dataset object and tokenize dataset
train_data, test_data = torchtext.legacy.data.TabularDataset.splits(
    path=dataset_dir, train='ratings_train.txt', test='ratings_test.txt', format='tsv',
    fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)

# Make Vocab
TEXT.build_vocab(train_data, min_freq=10)  # todo: nt - 10058, posix - 10070
LABEL.build_vocab(train_data)

# Define number of words and number of labels in the 'word vocabulary'
vocab_size = len(TEXT.vocab)
n_classes = 2

# Split validation data by 8:2
train_data, val_data = train_data.split(split_ratio=0.8)


# Make iterator-getter
def get_iterator(dataset):
    return torchtext.legacy.data.Iterator(dataset, batch_size=BATCH_SIZE, shuffle=True)


# Summarize dataset
print("DATASET INFORMATION")
print(
    "[train]: %d [val]: %d [test]: %d [words]: %d [class]: %d\n" %
    (len(train_data), len(val_data), len(test_data), vocab_size, n_classes)
)


#
# RNN Model implement
#

class RNN(nn.Module):
    """RNN model implement for Sentiment-analysis"""

    # Constructor
    # Register layers in constructor
    def __init__(
            self,
            num_vocab,
            num_classes,
            num_layers=1,
            embed_size=128,
            hidden_size=256,
            dropout_p=0.2
    ):
        nn.Module.__init__(self)
        self.num_layers = num_layers  # number of layer
        self.hidden_size = hidden_size  # hidden layer dimension
        self.embed = nn.Embedding(num_vocab, embed_size)  # n_vocab = number of words in the vocab. /
        # embed_size = dimension of embedding
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)  # For classification

    # Implement forward
    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))  # Initialize hidden state
        x, _ = self.rnn(x, h_0)  # [batch_size, sequence length, hidden_dim]
        h_t = x[:, -1, :]  # [batch_size, hidden_dim]
        self.dropout(h_t)
        return self.out(h_t)  # [batch_size, hidden_dim] -> [batch_size, n_classes]

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.num_layers, batch_size, self.hidden_size).zero_()

    # Backward will be implemented automatically,
    # by the pytorch framework itself.


#
# LSTM Model implement
#


class LSTM(nn.Module):
    """LSTM model implement for Sentiment-analysis"""

    # Constructor
    # Register layers in constructor
    def __init__(
            self,
            num_vocab,
            num_classes,
            num_layers=1,
            embed_size=128,
            hidden_size=256,
            dropout_p=0.2
    ):
        nn.Module.__init__(self)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_vocab, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.embed(x)

        # Initialize the hidden_state and cell_state
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))  # [batch_size, sequence length, hidden_dim]
        ht = out[:, -1, :]  # [batch_size, hidden_dim]
        self.dropout(ht)

        return self.out(ht)  # [batch_size, hidden_dim] -> [batch_size, n_classes]

    # Backward will be implemented automatically,
    # by the pytorch framework itself.


#
# Train / Test / Evaluate function implements
#


def train_model(
        model,
        train_iter,
        optimizer,
        log_hook=None,
        log_interval=50
):
    """
    Training function with using model, data loader, optimizer.

    :param model: torch.nn.Module
    :param train_iter:
    :param optimizer: torch.optim.Optimizer
    :param log_hook: (optional) when logging into stdout or gui, you can supply
                    logging hook callable, or class that has 'train' attribute
    :param log_interval: (optional) supply when using log hook. default is 10

    :return: None
    """
    # Set train mode
    model.train()
    # Iteration
    for iteration, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        # Zero grad
        optimizer.zero_grad()
        # Calculate with model and evaluate (forward propagation)
        output = model(x)  # Calculate by calling model
        loss = F.cross_entropy(output, y)  # Evaluate
        # Optimize model by gradient (backward propagation)
        loss.backward()  # get gradient with backward propagation
        optimizer.step()  # step: optimize weight params with gradient
        # Log train information
        if iteration % log_interval == 0 and log_hook is not None:
            getattr(log_hook, 'train', log_hook)(loss, iteration)


@torch.no_grad()  # stop autograd progress
def evaluate_model(
    model, val_iter,
    log_hook=None
):
    """
    Evaluating function with using model, data loader.

    :param model: torch.nn.Module
    :param val_iter:
    :param log_hook: (optional) when logging into stdout or gui, you can supply
                    logging hook callable, or class that has 'test' attribute

    :return: None
    """
    # Set test mode
    model.eval()
    # Initialize values
    total_loss, corrects = 0., 0
    # Iteration
    for batch in val_iter:
        # Convert device
        x, y = batch.text.to(device), batch.label.to(device)
        # Calculate with model
        output = model(x)
        # Append test-loss value
        total_loss += F.cross_entropy(output, y, reduction='sum').item()
        # Add accuracy count
        corrects += (output.max(1)[1].view(y.size()).data == y.data).sum()  # noqa
    # Calculate average test loss
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = corrects / size
    # Convert torch tensor to float
    if hasattr(avg_loss, 'item'):
        avg_loss = avg_loss.item()
    if hasattr(avg_accuracy, 'item'):
        avg_accuracy = avg_accuracy.item()
    # Log test information
    if log_hook is not None:
        getattr(log_hook, 'evaluate', log_hook)(avg_loss, avg_accuracy)
    return avg_loss, avg_accuracy


#
# Graph functions
#

def draw_figure(
        list_of_x,
        list_of_result_dict,
        total_epochs,
        name_of_x,
        title=None,
        verbose_name_of_x=None,
        save=False, dst=None,
):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 5)
    verbose_name_of_x = verbose_name_of_x or name_of_x

    # Parse inputs
    epoch_key = sorted({1, *range(5, total_epochs, 5), total_epochs})
    for key in epoch_key:
        lost_y, accuracy_y = [], []
        for result_dict in list_of_result_dict:
            lost_y.append(result_dict['loss'][key - 1])  # index
            accuracy_y.append(result_dict['accuracy'][key - 1] * 100)  # index
        ax1.plot(list_of_x, lost_y, label=f'epoch = {key}')
        ax2.plot(list_of_x, accuracy_y, label=f'epoch = {key}')

    # Draw evaluation plot
    ax1.set_title(f"Loss - {name_of_x} Graph")
    ax1.set_xlabel(verbose_name_of_x)
    ax1.set_ylabel("Loss value")
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax1.legend()

    ax2.set_title(f"Accuracy - {name_of_x} Graph")
    ax2.set_xlabel(verbose_name_of_x)
    ax2.set_ylabel("Accuracy value (%)")
    ax2.set_ylim([0, 100])
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.legend()

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')

    if save:
        plt.savefig(dst, bbox_inches='tight')

    plt.show()


#
# Actual running model-learn function implement
#

def run_model_learning(
        epoch=20,
        model_class=RNN,
        model_params=None,
        optimizer_class=torch.optim.Adam,
        learning_rate=1e-3,
        verbose=False,
        filename=None,
        **_model_params
):
    """
    ACTUAL function that executes model learning, by given epoch and train data ratio.

    :param epoch: (int) epochs
    :param model_class: model class
    :param model_params: (dict) model parameters
    :param optimizer_class: optimizer class
    :param learning_rate: (float) learning rate
    :param verbose: (bool) verbosity. with turning it on, you can view learning logs.
    :param filename: (str) In this path name, model's weight parameter will be saved.

    :return: trained-model.
    """

    verbose = verbose and reporthook is not None

    # Prepare datasets
    train_iter, val_iter = get_iterator(train_data), get_iterator(val_data)

    # Initialize model
    kwargs = dict(
        num_vocab=vocab_size,
        num_classes=n_classes,
    )
    kwargs.update(model_params or {})
    kwargs.update(_model_params)
    model = model_class(**kwargs).to(device)

    # Initialize optimizer
    optimizer = optimizer_class(
        model.parameters(), lr=learning_rate
    )

    if verbose:
        print(f"\n<Start Learning> total {epoch} epochs", end='\n\n')

    os.makedirs(snapshot_dir, exist_ok=True)
    processing_fn = os.path.join(snapshot_dir, '_processing.pt')
    best_val_loss = None

    # Do each epoch
    for index in range(1, epoch + 1):

        rph = reporthook.LearningReporthook(index, train_iter, val_iter) if verbose else None

        # Train model.
        train_model(
            model, train_iter, optimizer, log_hook=rph
        )

        l, a = evaluate_model(
            model, val_iter, log_hook=rph
        )

        # Save the model having the smallest validation loss
        if not best_val_loss or l < best_val_loss:
            torch.save(model.state_dict(), processing_fn)
            best_val_loss = l

    if verbose:
        print(f"\n<Stop Learning> Least loss: {best_val_loss}", end='\n\n')

    model.load_state_dict(torch.load(processing_fn))
    os.remove(processing_fn)

    if filename is not None:
        torch.save(model.state_dict(), os.path.join(snapshot_dir, filename))

    return model

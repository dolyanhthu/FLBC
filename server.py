from omegaconf import DictConfig
from typing import Dict
import flwr as fl
from model import get_model

from typing import Tuple, List
from flwr.common import Metrics

VERBOSE = 0

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            'lr': config.lr,
            'momentum': config.momentum,
            'local_epochs': config.local_epochs
        }
    
    return fit_config_fn

def get_evaluate_fn(test_set):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""
    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model()  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_set, verbose=VERBOSE)
        return loss, {"accuracy": accuracy}

    return evaluate
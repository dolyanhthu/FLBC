import flwr as fl

VERBOSE = 0
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_set, test_set, model) -> None:
        # Create model
        self.model = model
        self.train_set = train_set
        self.test_set = test_set

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_set, epochs=10, verbose=VERBOSE)
        return self.model.get_weights(), len(self.train_set), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.test_set, verbose=VERBOSE)
        return loss, len(self.test_set), {"accuracy": acc}
    
def generate_client_fn(trainloaders, valloaders):
    def client_fn(cid: str) -> fl.client.Client:
        client = FlowerClient(trainloaders=trainloaders[int(cid)], 
                            valloaders=valloaders[int(cid)])
        return client.to_client()
    
    return client_fn
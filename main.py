import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset

@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(cfg: DictConfig):

    # 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg=cfg))

    # 2. Prepare datasets
    train_loaders, validation_loaders, test_loaders = prepare_dataset(cfg.num_clients, cfg.batch_size)

    print(len(train_loaders), len(train_loaders[0].dataset))

if __name__ == "__main__":
    main()
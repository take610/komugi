import argparse

import torch
from models.networks.efficient_det_5 import get_net
from models.fitter import Fitter
from scripts.dataloder import get_dataloader
from scripts.load_config import load_config


def run_training(path_data, config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = get_net()
    net.to(device)

    train_loader, val_loader = get_dataloader(path_data, config)

    fitter = Fitter(model=net, device=device, config=config)
    fitter.fit(train_loader, val_loader)


def main(args):
    config = load_config(args.config)
    run_training(args.data, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/default.yaml", type=str)
    parser.add_argument("--data", default="../input/global-wheat-detection", type=str)
    args = parser.parse_args()
    main(args)

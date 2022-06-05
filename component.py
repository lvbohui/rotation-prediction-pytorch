import os
import json
import logging

from distutils.version import LooseVersion
from torch.utils.tensorboard import SummaryWriter

from arguments import parse_args
from train import get_transform_dict, train

from datasets.loaders import create_loaders
from datasets.component_dataset import ComponentDataset
from utils.train import model_init
from utils.misc import get_save_path, initialize_logger, seed, save_args
from models.model_factory import MODEL_GETTERS


logger = logging.getLogger()


def main(args, save_path: str):
    """
    Main function that sets up and starts the RotNet training
    """
    writer = SummaryWriter(save_path)

    # Get dictionary which contains train transforms (both for labeled and unlabeled batches) 
    # as well as the transform for the validation and test set
    transform_dict = get_transform_dict(args)

    # Get torch dataset objects from specified dataset
    train_dir = "data/component/train"
    valid_dir = "data/component/valid"
    test_dir = "data/component/valid"
    train_set = ComponentDataset(train_dir, transform=transform_dict["train"])
    validation_set = ComponentDataset(valid_dir, transform=transform_dict["test"])
    test_set = ComponentDataset(test_dir, transform=transform_dict["test"])

    # Get loaders for the labeled and unlabeled train set as well as the validation and test set
    args.iters_per_epoch = 2 # (len(train_set) // args.batch_size) + 1
    train_loader, validation_loader, test_loader = create_loaders(
        args,
        train_set,
        validation_set,
        test_set,
        args.batch_size,
        total_iters=args.iters_per_epoch,
    )

    # Print and log dataset stats
    logger.info("-------- Starting Unsupervised Rotation Prediction Training --------")
    logger.info("\t- Train set: {}".format(len(train_set)))
    logger.info("\t- Validation set: {}".format(len(validation_set)))
    logger.info("\t- Test set: {}".format(len(test_set)))
    logger.info("\t- Train dir: {}".format(train_dir))

    logger.info("-------- MODEL --------")
    args.num_classes = 4
    model = MODEL_GETTERS[args.model](num_classes=args.num_classes)
    model.apply(model_init)
    num_params = sum([p.numel() for p in model.parameters()])
    logger.info("\t- Number of parameters: {}".format(num_params))
    logger.info("\t- Number of target classes: {}".format(args.num_classes))

    # Start rotation prediction training
    train(
        args,
        model,
        train_loader,
        validation_loader,
        test_loader,
        writer,
        save_path=save_path
    )
    save_args(args, save_path)


if __name__ == '__main__':
    # Read command line arguments
    args = parse_args()

    save_path = get_save_path(args)

    initialize_logger(save_path)
    args.seed = seed(args.random_seed, args.seed)
    logger.info("Seed is set to {}".format(args.seed))
    main(args, save_path)

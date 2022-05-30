import argparse
from cgi import test
import logging
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from distutils.version import LooseVersion
from utils.metrics import *
from utils.train import rotate_tensors, ModelWrapper, NINWrapper
from utils.eval import AverageMeterSet
from datasets.custom_datasets import CustomSubset
from datasets.component_dataset import ComponentDataset
from models.network_in_network import NetworkInNetwork

import cv2
import numpy as np

logger = logging.getLogger()

print(torch.cuda.is_available())

def evaluate(
        args,
        eval_loader: DataLoader,
        model: nn.Module,
        epoch: int,
        descriptor: str = "Test",
):
    """
    Evaluates current model based on the provided evaluation dataloader

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    eval_loader: torch.utils.data.DataLoader
        DataLoader objects which loads batches of evaluation dataset
    model: nn.Module
        Current model which should be evaluated on prediction task
    epoch: int
        Current epoch which is used for progress bar logging if enabled
    descriptor: str
        Descriptor which is used for progress bar logging if enabled

    Returns
    -------
    eval_tuple: namedtuple
        NamedTuple which holds all evaluation metrics such as accuracy, precision, recall, f1
    """
    meters = AverageMeterSet()

    model.eval()
    
    if args.device == 'cuda':
        model.cuda()
    else:
        model.cpu()


    if args.pbar:
        p_bar = tqdm(range(len(eval_loader)))

    with torch.no_grad():
        for i, (inputs, _) in enumerate(eval_loader):   

            inputs, rot_targets = rotate_tensors(inputs) # rotation and connect image

            inputs = inputs.to(args.device).float()
            rot_targets = rot_targets.to(args.device)

            # Output
            logits = model(inputs)
            loss = F.cross_entropy(logits, rot_targets, reduction="mean")

            # Compute metrics
            (top1,) = accuracy(logits, rot_targets, topk=(1,))
            meters.update("loss", loss.item(), len(inputs))
            meters.update("top1", top1.item(), len(inputs))

            if args.pbar:
                p_bar.set_description(
                    "{descriptor}: Epoch: {epoch:4}. Iter: {batch:4}/{iter:4}. Class loss: {cl:4}. Top1: {top1:4}.".format(
                        descriptor=descriptor,
                        epoch=epoch + 1,
                        batch=i + 1,
                        iter=len(eval_loader),
                        cl=meters["loss"],
                        top1=meters["top1"],
                    )
                )
                p_bar.update()
    if args.pbar:
        p_bar.close()
    logger.info(" * Prec@1 {top1.avg:.3f}".format(top1=meters["top1"]))
    return meters["loss"].avg, meters["top1"].avg


def parse_args():
    parser = argparse.ArgumentParser(description='RotNet evaluation')

    parser.add_argument('--run-path', type=str, help='path to RotNet run which should be evaluated.')
    parser.add_argument('--data-dir', default='./data', type=str, help='path to directory where datasets are saved.')
    parser.add_argument('--checkpoint-file', default='', type=str, help='name of .tar-checkpoint file from which model is loaded for evaluation.')
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'], help='device (cpu / cuda) on which evaluation is run.')
    parser.add_argument('--pbar', action='store_true', default=False, help='flag indicating whether or not to show progress bar for evaluation.')
    return parser.parse_args()


if __name__ == '__main__':
    import os
    from utils.misc import load_dataset_indices, load_args, load_state
    from augmentation.augmentations import get_normalizer
    from datasets.datasets import get_datasets, get_base_sets
    from models.model_factory import MODEL_GETTERS

    args = parse_args()
    args.device = torch.device(args.device)

    # Load arguments of run to evaluate
    run_args = load_args(args.run_path)


    test_dir = "data/sink/valid/up"
    test_set = ComponentDataset(test_dir)
    print("Number of test samples:{}".format(len(test_set)))

    # Get loaders for the labeled and unlabeled train set as well as the validation and test set
    args.iters_per_epoch = 10 # (len(train_set) // args.batch_size) + 1
    test_loader = DataLoader(test_set,
                             batch_size=run_args.batch_size,
                             num_workers=run_args.num_workers,
                             shuffle=False,
                             pin_memory=run_args.pin_memory,
    )

    # Load trained model from specified checkpoint .tar-file containing model state dict
    model = MODEL_GETTERS[run_args.model](num_classes=run_args.num_classes)

    if args.checkpoint_file:
        saved_state = load_state(os.path.join(args.run_path, args.checkpoint_file), map_location=args.device)
    else:
        checkpoint_file = next(filter(lambda x: x.endswith('.tar'), sorted(os.listdir(args.run_path), reverse=True)))
        saved_state = load_state(os.path.join(args.run_path, checkpoint_file), map_location=args.device)

    model.load_state_dict(saved_state['model_state_dict'])
    loss, top1_acc = evaluate(run_args, test_loader, model, saved_state['epoch'])

    print(' RotNet EVALUATION '.center(50, '-'))
    print(f'\t - Dataset {run_args.dataset}')
    print(f'\t - Model {run_args.model}')
    print(f'\t - Test metrics:')
    print(f'\t\tloss: {loss}')
    print(f'\t\ttop1_accuracy: {top1_acc}')

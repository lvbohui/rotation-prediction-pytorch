import argparse
import logging

import torch
from torch.utils.data import DataLoader

from distutils.version import LooseVersion
from utils.metrics import *
from utils.train import rotate_tensors
from datasets.component_dataset import ComponentDataset

import cv2
import numpy as np

import os
from utils.misc import load_args, load_state
from models.model_factory import MODEL_GETTERS

logger = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(description='RotNet evaluation')

    parser.add_argument('--run-path', type=str, required=True, help='path to RotNet run which should be evaluated.')
    parser.add_argument('--data-dir', default='./data', type=str, help='path to directory where datasets are saved.')
    parser.add_argument('--checkpoint-file', default='', type=str, help='name of .tar-checkpoint file from which model is loaded for evaluation.')
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'], help='device (cpu / cuda) on which evaluation is run.')
    parser.add_argument('--pbar', action='store_true', default=False, help='flag indicating whether or not to show progress bar for evaluation.')
    return parser.parse_args()


class DefaultPredictor(object):
    def __init__(self, model):
        self.model = model.eval()
        self.model.cuda()

    def __call__(self, image):
        image, _ = rotate_tensors(image)

        image.to("cuda")
        output = self.model(image)
        return output

if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device(args.device)

    # Load arguments of run to evaluate
    run_args = load_args(args.run_path)

    # Load trained model from specified checkpoint .tar-file containing model state dict
    model = MODEL_GETTERS[run_args.model](num_classes=run_args.num_classes)

    if args.checkpoint_file:
        saved_state = load_state(os.path.join(args.run_path, args.checkpoint_file), map_location=args.device)
    else:
        checkpoint_file = next(filter(lambda x: x.endswith('.tar'), sorted(os.listdir(args.run_path), reverse=True)))
        saved_state = load_state(os.path.join(args.run_path, checkpoint_file), map_location=args.device)

    model.load_state_dict(saved_state['model_state_dict'])

    model.eval()
    model.cuda()
    # Set index to category name
    index2name = {
        0: "up",
        1: "left",
        2: "down",
        3: "right"
    }

    # Load dataset
    test_image_dir = "../Image-classification/data/sink/valid/left"
    test_set = ComponentDataset(test_image_dir, transform=None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            # image = image.to(args.device)
            
            image, rot_targets = rotate_tensors(image) # rotation and connect image
            image = image.to(args.device).float()

            # Output
            output = model(image)

            predict_result = output[0].cpu().numpy()
            predict_name = index2name[np.argmax(predict_result)]
            print(predict_name)

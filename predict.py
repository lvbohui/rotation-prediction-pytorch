import argparse
import json
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch
from models.model_factory import MODEL_GETTERS


def rotate_tensors(batch):
    rotated_samples = torch.cat(
        [
            batch,
            batch.transpose(2, 3).flip(2),
            batch.flip(2).flip(3),
            batch.transpose(2, 3).flip(3),
        ]
    )
    return rotated_samples


class RotationPredictor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        time1 = time.time()
        # Load trained model from specified checkpoint .tar-file containing model state dict
        self.model = MODEL_GETTERS[self.cfg.model](num_classes=self.cfg.num_classes)

        checkpoint_file = self.cfg.checkpoint_file
        saved_state = torch.load(checkpoint_file, map_location=self.cfg.device)
        self.model.load_state_dict(saved_state['model_state_dict'])

        self.model.eval()
        self.model.to(self.cfg.device)
        print("Load model time:{}".format(time.time()-time1))
        self.image_dim = (32, 32)
        self.index2name = {
                            0: "up",
                            1: "left",
                            2: "down",
                            3: "right"
        }

    def __call__(self, image):
        image = cv2.resize(image, self.image_dim)

        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2, 0, 1) # Adjust [H, W, C] to [C, H, W]
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension, [B, C, H, W]

        time2 = time.time()
        image = rotate_tensors(image_tensor) # Rotate and connect image_tensor
        image = image.to(self.cfg.device).float() # Convert to cuda and float
        print("Rotation time:{}".format(time.time()-time2))

        time3 = time.time()
        output = self.model(image) # Predict
        print("Prediction time:{}".format(time.time()-time3))
        predict_result = output.detach().cpu().numpy()
        predict_name = self.index2name[np.argmax(predict_result[0])]

        return predict_name


def parse_args():
    parser = argparse.ArgumentParser(description='RotNet prediction')

    parser.add_argument('--config-file', type=str, required=True, 
                        help='path to RotNet config file.')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='path to image to predict.')
    parser.add_argument('--checkpoint-file', type=str, default="out/rotnet_training/component/last_model.tar",
                        help='path to checkpoint file.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load arguments of run to predict
    run_args = SimpleNamespace(**json.load(open(args.config_file)))
    run_args.checkpoint_file = args.checkpoint_file

    # Get prediction image
    image = cv2.imread(args.image_dir)
    
    predictor = RotationPredictor(run_args)
    predict_name = predictor(image)
    print(predict_name)

"""
Rotation prediction handler
"""
import base64
import io
import json
import logging
import os
from types import SimpleNamespace

from PIL import Image

import numpy as np
import orjson
import torch
from predict import RotationPredictor
from ts.torch_handler.base_handler import BaseHandler


os.makedirs("logs", exist_ok=True)
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs/rotnet_handler.log')
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger(__name__)
logger.info("Rotation prediction service started")

class ModelHandler(BaseHandler):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.config_file = "rotation_component.json"
        self.model_file = "rotation_0.1.pth"

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        logger.info("initializing starting")
        properties = context.system_properties
        logger.info(f"Get system info:{properties}")

        cfg = SimpleNamespace(**json.load(open(self.model_file)))
        cfg.checkpoint_file = self.model_file

        logger.info("File {} exists {}".format(cfg.checkpoint_file, str(os.path.exists(cfg.checkpoint_file))))
        logger.info("File {} exists {}".format(self.config_file, str(os.path.exists(self.config_file))))

        # set the testing threshold for this model
        logger.info("torch.__version__:{}".format(torch.__version__))
        logger.info("CUDA status:{}".format(torch.cuda.is_available()))

        use_gpu = False if os.getenv("CUDA_VISIBLE_DEVICES") == "-1" else True
        cfg.device = "cuda:{}".format(properties.get("gpu_id")) \
                            if properties.get("gpu_id") != None and use_gpu != False else "cpu"

        logger.info(f"MODEL.DEVICE:{cfg.device}")
        logger.info("properties.gpu_id:{}".format(properties.get("gpu_id")))

        logger.info("predictor built")
        self.predictor = RotationPredictor(cfg)

        logger.info("predictor built on initialize")

        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True
        logger.info("initialized")

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        # Take the input data and pre-process it make it inference ready
        images = []
        # batch is a list of requests
        for request in batch:
            # each item in the list is a dictionary with a single body key, get the body of the request
            image = request.get("body") or request.get("data")

            # get our image
            if isinstance(image, str):
                # if the image is encoded with base64
                image = base64.b64decode(image)
            elif isinstance(image, (bytearray, bytes)):
                # if the image is a string of bytesarray.
                image = Image.open(io.BytesIO(image))
                image = np.array(image)
            else:
                # if the image is a list
                image = torch.FloatTensor(image)
            # add the image to our list
            logger.info("Image type:\n{}".format(type(image)))
            images.append(image)
        return images

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        outputs = []
        for image in model_input:
            # run our predictions
            output = self.predictor(image)
            outputs.append(output)
            # clear cache
            torch.cuda.empty_cache()
        return outputs


    def postprocess(self, inference_output):

        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        responses = orjson.dumps(inference_output)
        return responses

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        # process the data through our inference pipeline
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        output = self.postprocess(model_out)
        return output

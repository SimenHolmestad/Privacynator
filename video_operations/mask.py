import numpy as np
from video_operations.base_video_operation import BaseVideoOperation
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()


class Mask(BaseVideoOperation):
    def __init__(self):
        self.initialize_detectron2_model()
        self.coco_classes = [2, 7]

    def do_operation(self, image, output_video):
        output_video.write(self.mask_coco_instances(image))

    def mask_coco_instances(self, image):
        output = self.detectron2_model(image)
        output_instances = output["instances"]

        output_classes = output_instances.pred_classes
        output_masks = output_instances.pred_masks

        for index, coco_class in enumerate(output_classes):
            if (coco_class in self.coco_classes):
                image = self.mask_image(image, output_masks[index])

        return image

    def mask_image(self, image, mask):
        mask = self.convert_mask_to_right_format(mask)
        inverted_mask = np.abs(mask - 1)
        return np.uint8(image * inverted_mask)

    def convert_mask_to_right_format(self, mask):
        numpy_mask = mask.numpy()
        width, height = numpy_mask.shape
        return numpy_mask.reshape((width, height, 1))

    def initialize_detectron2_model(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for the model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.detectron2_model = DefaultPredictor(cfg)

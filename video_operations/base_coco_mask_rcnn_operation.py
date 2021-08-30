from abc import ABC
from video_operations.base_video_operation import BaseVideoOperation
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class BaseCocoMaskRcnnOperation(BaseVideoOperation, ABC):
    def __init__(self):
        self.initialize_detectron2_model()
        self.coco_classes = [2, 7]

    def initialize_detectron2_model(self):
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for the model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.detectron2_model = DefaultPredictor(self.cfg)

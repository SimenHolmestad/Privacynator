from video_operations.base_video_operation import BaseVideoOperation
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()


class RejectFramesWithPii(BaseVideoOperation):
    def __init__(self):
        self.initialize_detectron2_model()

    def do_operation(self, image, output_video):
        if self.image_does_not_contain_pii(image):
            output_video.write(image)

    def image_does_not_contain_pii(self, image):
        outputs = self.detectron2_model(image)
        print("-----------------------------------------------")
        print(outputs["instances"].pred_classes)
        print(outputs["instances"])
        # TODO: Implement this
        return True

    def initialize_detectron2_model(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for the model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.detectron2_model = DefaultPredictor(self.cfg)

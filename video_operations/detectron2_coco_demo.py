from video_operations.base_video_operation import BaseVideoOperation
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()


class Detectron2CocoDemo(BaseVideoOperation):
    def __init__(self):
        self.initialize_detectron2_model()

    def do_operation(self, image, output_video):
        output_video.write(self.do_demo_on_image(image))

    def do_demo_on_image(self, image):
        outputs = self.detectron2_model(image)
        return self.draw_model_outputs_on_image(image, outputs)

    def draw_model_outputs_on_image(self, image, outputs):
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(outputs["instances"])
        return out.get_image()[:, :, ::-1]

    def initialize_detectron2_model(self):
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for the model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.detectron2_model = DefaultPredictor(self.cfg)

from video_operations.base_coco_mask_rcnn_operation import BaseCocoMaskRcnnOperation
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class Detectron2CocoDemo(BaseCocoMaskRcnnOperation):
    def do_operation(self, image, output_video):
        output_video.write(self.do_demo_on_image(image))

    def do_demo_on_image(self, image):
        outputs = self.detectron2_model(image)
        return self.draw_model_outputs_on_image(image, outputs)

    def draw_model_outputs_on_image(self, image, outputs):
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

from video_operations.base_coco_mask_rcnn_operation import BaseCocoMaskRcnnOperation


class RejectFramesWithPii(BaseCocoMaskRcnnOperation):
    def do_operation(self, image, output_video):
        if self.image_does_not_contain_pii(image):
            output_video.write(image)

    def image_does_not_contain_pii(self, image):
        output = self.detectron2_model(image)
        return not self.output_contains_coco_instances(output, self.coco_classes)

    def output_contains_coco_instances(self, output, coco_classes):
        output_classes = output["instances"].pred_classes
        for coco_class in coco_classes:
            if coco_class in output_classes:
                return True
        return False

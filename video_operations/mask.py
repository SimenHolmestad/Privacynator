import numpy as np
from video_operations.base_coco_mask_rcnn_operation import BaseCocoMaskRcnnOperation


class Mask(BaseCocoMaskRcnnOperation):
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

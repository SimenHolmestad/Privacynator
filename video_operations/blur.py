import functools
import numpy as np
from video_operations.base_coco_mask_rcnn_operation import BaseCocoMaskRcnnOperation
from scipy.ndimage.filters import gaussian_filter


class Blur(BaseCocoMaskRcnnOperation):
    def do_operation(self, image, output_video):
        output_video.write(self.blur_coco_instances(image))

    def blur_coco_instances(self, image):
        masks_with_coco_classes = self.get_coco_masks_for_image(image)
        if len(masks_with_coco_classes) == 0:
            return image
        combined_masks = self.combine_masks(masks_with_coco_classes)
        return self.blur_masked_parts_of_image(image, combined_masks)

    def blur_masked_parts_of_image(self, image, mask):
        blurred_image = gaussian_filter(image, sigma=5)
        mask = self.convert_mask_to_right_format(mask)
        inverted_mask = np.abs(mask - 1)

        image_with_masked_parts_removed = image * inverted_mask
        blurred_masked_parts_of_image = blurred_image * mask
        return np.uint8(image_with_masked_parts_removed + blurred_masked_parts_of_image)

    def convert_mask_to_right_format(self, mask):
        numpy_mask = mask.numpy()
        width, height = numpy_mask.shape
        return numpy_mask.reshape((width, height, 1))

    def get_coco_masks_for_image(self, image):
        output = self.detectron2_model(image)
        output_instances = output["instances"]

        output_classes = output_instances.pred_classes
        output_masks = output_instances.pred_masks

        return self.extract_masks_with_coco_classes(
            output_masks,
            output_classes
        )

    def extract_masks_with_coco_classes(self, masks, classes):
        output_masks = []
        for index, coco_class in enumerate(classes):
            if (coco_class in self.coco_classes):
                output_masks.append(masks[index])

        return output_masks

    def combine_masks(self, masks):
        return functools.reduce(lambda a, b: a + b, masks)

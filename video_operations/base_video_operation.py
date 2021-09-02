from abc import ABC, abstractmethod


class BaseVideoOperation(ABC):
    """Abstract class for video operations"""

    def __init__(self, coco_classes):
        self.coco_classes = coco_classes

    @abstractmethod
    def do_operation(self, image, output_video):
        """Writes 0, 1 or more images to the output video stream based on input image

        An image can be written to the output video stream as follows:
        ```
        output_video.write(image)
        ```
        """
        pass

from video_operations.reject_frames_with_pii import RejectFramesWithPii
from video_operations.detectron2_coco_demo import Detectron2CocoDemo
from video_operations.mask import Mask
from video_operations.blur import Blur

video_operation_options = {
    "reject_frames": RejectFramesWithPii,
    "demo": Detectron2CocoDemo,
    "mask": Mask,
    "blur": Blur,
}


class VideoOperationNotFoundError(Exception):
    pass


def get_video_operation_names():
    """Returns a list with the names of the available video operations"""
    return list(video_operation_options.keys())


def get_video_operation_class_by_name(name):
    if name not in get_video_operation_names():
        raise VideoOperationNotFoundError()

    return video_operation_options[name]

# Privacynator
Privacynator is an utility for removing Personally Identifiable Information (PII) from videos. It is primarily meant for videos captured from cars.

Currently, Privacynator uses the Detectron2 model from facebook research (<https://github.com/facebookresearch/detectron2>).

# Usage
Privacynator is run as follows:
``` sh
python3 privatize.py <video-operation> <input-file-path> <output-file-path>
```

There is also quite some options to add. For example, if you want to:
- Overvrite (`-o`) the output file if it exists
- Start (`-s`) from frame 68
- Limit (`-l`) video to the 419 next frames

Your command would be
``` sh
python3 privatize.py -o -s 68 -l 419 <video-operation> <input-file-path> <output-file-path>
```

It is also possible to use different Coco classes (`-c`) for the application. So if you for some reason would like to blur all frisbees and giraffes in your video, you can run the command:

``` sh
python3 privatize.py blur <input-file-path> <output-file-path> -c 33 24
```

A list of all the Coco classes can be found by following [this link](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/). **Note:** The Coco class numbers in the link are one-indexed, while the input Coco class numbers to privatize to be 0-indexed.

# Video operations
There are several possible techniques to remove PII from a video, and Privacynator supports some of them. The available methods are each implemented as its own "video operation".

The currently available video operations are:
## Reject
Rejects frames from the input video containing PII. This means your output video probably will be a lot shorter.

## Mask
Create black masks over regions containing PII in the video.

## Blur
Blur areas containing PII in the video.

## Demo
Does a semantic segmentation on every frame using Detectron2 and outputs a visualisation of the classes. This is sometimes nice for debugging.

## Adding a new video operation
A video operation is a fairly simple thing. It takes an image as input and ouputs 0, 1 or more images to an output based on the content of the input image.

To add a new video operation, create a subclass of the `BaseVideoOperation` class and implement the function `do_operation`. Then, register the new video operation in the `video_operation_options` dict in the file `video_operation_options.py`.

The `BaseVideoOperation` class can be found at [video_operations/base_video_operation.py](video_operations/base_video_operation.py).


# Testing out Privacynator
An example of a video to privatize can be found at <https://www.pexels.com/video/video-of-travel-854669>.

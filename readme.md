# Privacynator
Privacynator is an utility for removing Personally Identifiable Information (PII) from videos. It is primarily meant for videos captured from cars.

Privacynator uses the Detectron2 model from facebook research: <https://github.com/facebookresearch/detectron2>.

# Installation
The installation process is not completely straight forward as it depends on the CUDA version of your system.
## Create a virtual environment (optional, but recommended)
To create a virtual environment with python, use the following command:
``` sh
python3 -m venv venv
```

Then, activate this environment by doing
``` sh
source ./venv/bin/activate
```

## Installing the correct dependencies
First, install Torch and Torchvision by following the instructions at <https://pytorch.org/>, making sure that your version of CUDA is supported.

Then, install Detectron2 by following instructions at <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>. **NOTE**: The Detectron2 version must match the PyTorch version!

Finally, install the rest of the dependencies by doing
``` sh
pip install -r requirements.txt
```

# Usage
To run Privacynator, use the follwoing command:
``` sh
python3 privatize.py <video-operation> <input-file-path> <output-file-path>
```

Where `video-operation` can be either `reject`, `blur`, `mask` or `demo`. More information on those can be found further down in the readme.

`privatize.py` also comes with several keyword arguments. For example, if you want to:
- Overwrite (`-o`) the output file if it exists
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

A list of all the Coco classes can be found by following [this link](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/). **Note:** The Coco class numbers in the link are one-indexed, while the input Coco class numbers to `privatize.py` has to be 0-indexed.

# Video operations
There are several possible techniques to remove PII from a video, and Privacynator supports some of them. The available methods are each implemented as its own "video operation".

The currently available video operations are:
## Reject
Rejects frames from the input video containing PII. This means your output video probably will be a lot shorter than the input video.

## Mask
Create black masks over regions containing PII in the video.

## Blur
Blur areas containing PII in the video.

## Demo
This video operation just runs Detectron2's demo visualization on every frame of the video. This can be nice for debugging if the other video operations makes weird results.

## Adding a new video operation
A video operation is fairly simple. It takes an image as input and ouputs 0, 1 or more images to an output based on the content of the input image.

To add a new video operation, create a subclass of the `BaseVideoOperation` class and implement the function `do_operation`. Then, register the new video operation in the `video_operation_options` dict in the file `video_operation_options.py`.

The `BaseVideoOperation` class can be found at [video_operations/base_video_operation.py](video_operations/base_video_operation.py).

# Testing out Privacynator
An example of a video to privatize can be found at <https://www.pexels.com/video/video-of-travel-854669>.

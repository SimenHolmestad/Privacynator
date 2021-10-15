import sys
import os
import argparse
import cv2
from tqdm import tqdm
from video_operation_options import get_video_operation_names, get_video_operation_class_by_name


def main():
    args = parse_command_line_args()
    convert_video_file(
        args.video_operation,
        args.input_file,
        args.output_file,
        args.start_from_frame,
        args.coco_classes,
        args.limit_to_frame
    )


def convert_video_file(video_operation_name, input_filename, output_filename, start_from_frame, coco_classes, limit_to_frame=None):
    input_video = cv2.VideoCapture(input_filename)
    input_video.set(cv2.CAP_PROP_POS_FRAMES, start_from_frame)
    output_video = create_output_video(input_video, output_filename)

    VideoOperationClass = get_video_operation_class_by_name(video_operation_name)
    video_operation = VideoOperationClass(coco_classes)
    do_operation_on_video(input_video, output_video, video_operation, limit_to_frame)

    input_video.release()
    output_video.release()


def do_operation_on_video(input_video, output_video, video_operation, limit_to_frame=None):
    num_frames_to_process = get_number_of_frames_to_process(input_video, limit_to_frame)

    for frame_number in tqdm(range(num_frames_to_process)):
        ret, image = input_video.read()
        if ret is False:
            break

        video_operation.do_operation(image, output_video)


def get_number_of_frames_to_process(input_video, limit_to_frame):
    num_frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit_to_frame and limit_to_frame < num_frames_in_video:
        return limit_to_frame
    return num_frames_in_video


def create_output_video(input_video, output_filename):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    framesize = (int(width), int(height))
    return cv2.VideoWriter(output_filename, fourcc, fps, framesize)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_operation", choices=get_video_operation_names(),
                        help="The video operation to be performed on the video")
    parser.add_argument("input_file",
                        help="Path to video to be converted using privatize")
    parser.add_argument("output_file",
                        help="Path to the output file")
    parser.add_argument("-o", "--overwrite_output_file",
                        action="store_true",
                        help="Overwrite output file if it exists")
    parser.add_argument("-l", "--limit_to_frame", type=int,
                        help="Limit conversion to specified amount of frames from start of input video")
    parser.add_argument("-s", "--start_from_frame", type=int, default=0,
                        help="Start conversion from specified frame in input video (0-indexed)")
    parser.add_argument("-c", "--coco_classes", type=int, nargs='+', default=[0, 1, 2, 3, 5, 7, 17, 26],
                        help="Coco classes to use when applying video operation")

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print("error:", args.input_file, "is not a file.")
        sys.exit()

    if os.path.isfile(args.output_file) and not args.overwrite_output_file:
        print("error:", args.output_file, "already exists")
        print("Use the -o argument to overwrite the output file")
        sys.exit()

    return args


if __name__ == '__main__':
    main()

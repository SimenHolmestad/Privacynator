import sys
import os
import argparse
import cv2


def main():
    args = parse_command_line_args()
    convert_video_file(
        args.input_file,
        args.output_file,
        args.limit_to_frame
    )


def convert_video_file(input_filename, output_filename, limit_to_frame=None):
    input_video = cv2.VideoCapture(input_filename)
    output_video = create_output_video(input_video, output_filename)

    frame_count = 0
    while(input_video.isOpened()):
        ret, image = input_video.read()
        if ret is False:
            break

        if is_valid_image(image):
            output_video.write(image)

        frame_count += 1
        if limit_to_frame and frame_count >= limit_to_frame:
            break

    input_video.release()
    output_video.release()


def is_valid_image(frame):
    return True


def create_output_video(input_video, output_filename):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    framesize = (int(width), int(height))
    return cv2.VideoWriter(output_filename, fourcc, fps, framesize)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help="Path to video to be converted using privatize")
    parser.add_argument("output_file",
                        help="Path to the output file")
    parser.add_argument("-o", "--overwrite_output_file",
                        action="store_true",
                        help="Overwrite output file if it exists")
    parser.add_argument("-l", "--limit_to_frame", type=int,
                        help="Limit conversion to specified amount of frames from start of input video")

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

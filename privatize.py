import sys
import os
import argparse
import torch
import torchvision.io


def main():
    args = parse_command_line_args()
    print("Privatizing file \"{}\" to \"{}\"".format(args.input_file, args.output_file))

    print("\nLoading file", args.input_file)
    reader = torchvision.io.VideoReader(args.input_file, "video")

    valid_frames = []

    for frame_number, frame in enumerate(reader):
        image = frame["data"]

        if is_valid_image(image):
            valid_frames.append(convert_image_to_single_movie_frame(image))

        if args.limit_frames and frame_number >= args.limit_frames:
            break

    output_frames = torch.cat(valid_frames, dim=0)
    torchvision.io.write_video(args.output_file, output_frames, fps=24)


def is_valid_image(frame):
    return True


def convert_image_to_single_movie_frame(image):
    hwc_image = torch.moveaxis(image, 0, 2)
    shape = hwc_image.shape
    new_shape = (1, shape[0], shape[1], shape[2])
    return hwc_image.reshape(new_shape)


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help="Path to video to be converted using privatize")
    parser.add_argument("output_file",
                        help="Path to the output file")
    parser.add_argument("-o", "--overwrite_output_file",
                        action="store_true",
                        help="Overwrite output file if it exists")
    parser.add_argument("-l", "--limit_frames", type=int,
                        help="Limit to specified amount of frames")

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

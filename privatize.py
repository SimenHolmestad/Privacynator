import sys
import os
import argparse
import cv2


def main():
    args = parse_command_line_args()
    cap = cv2.VideoCapture(args.input_file)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output_file, fourcc, 20.0, (1920, 1080))

    frame_count = 0
    while(cap.isOpened()):
        ret, image = cap.read()
        if ret is False:
            break

        if is_valid_image(image):
            out.write(image)

        frame_count += 1
        if args.limit_frames and frame_count >= args.limit_frames:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def is_valid_image(frame):
    return True


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

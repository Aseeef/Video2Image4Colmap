import argparse
import os
import time

import cv2 as cv



def compare_images(image1, image2):
    diff = cv.absdiff(image1, image2)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    mean, _ = cv.meanStdDev(diff)
    return mean[0]


def get_blurry_amount(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image
    laplacian = cv.Laplacian(gray, cv.CV_64F)

    cv.putText(laplacian, f'Blurry: {laplacian.var()}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    im_show_debug(laplacian, 'Laplacian', True)

    # If the variance is below the threshold, the image is blurry
    return laplacian.var()


def get_program_params() -> argparse.Namespace:
    # Get the program parameters
    parser = argparse.ArgumentParser(
        description='Command line python script to extract a sequence of the least blurry frames \
                    from a video file for colmap processing'
    )
    parser.add_argument('arg1', type=str, default='input.mp4',
                        help='Path to the video file (default: input.mp4)')
    parser.add_argument('--blur-threshold', type=int, default=33,
                        help='The amount of blur allowed in a frame (lower=more blur allowed) (default: 33)')
    parser.add_argument('--output-format', type=str, default="output/output_%04d.png",
                        help='The output format for the images (default: output_%04d.png)')
    parser.add_argument('--max-diff', type=int, default=30,
                        help='The largest difference error allowed between frames (default: 100)')
    parser.add_argument('--min-diff', type=int, default=15,
                        help='The smallest difference error allowed between frames (default: 20)')
    parser.add_argument('--resize-scale', type=float, default=1.0,
                        help='Whether to resize the output images (default: 1.0 (no resize))')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Show intermediate images for debugging')
    return parser.parse_args()


def write_and_show_image(image, output_format, img_num, debug):
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_format), exist_ok=True)
    # write the image
    cv.imwrite(output_format % img_num, image)
    # show the image if debugging
    im_show_debug(image, 'Wrote Image', debug)


def im_show_debug(image, frame_name, debug):
    if debug:
        cv.imshow(frame_name, image)
        cv.waitKey(0)

def main():
    arguments = get_program_params()

    video_source = arguments.arg1
    blur_threshold = arguments.blur_threshold
    output_format = arguments.output_format
    max_diff = arguments.max_diff
    min_diff = arguments.min_diff
    resize_scale = arguments.resize_scale
    debug = arguments.debug

    capture = cv.VideoCapture(video_source)
    prev_frame = None
    frames_to_consider = []
    img_num = 0
    while True:
        success, frame = capture.read()
        # If the frame is not read, then we have reached the end of the video
        if not success:
            break
        # Resize the frame if needed
        if resize_scale != 1.0:
            frame = cv.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        # Show the current frame if debugging
        im_show_debug(frame, 'Current Frame', debug)
        # Determine if the frame is blurry and by how much
        blur_amnt = get_blurry_amount(frame)
        is_blurry = blur_amnt < blur_threshold
        # If the frame is blurry, show it if debugging
        if is_blurry:
            im_show_debug(frame, 'Blurry Frame (Excluded)', debug)
        # If this is the first non-blurry frame, write it out
        elif not is_blurry and prev_frame is None:
            prev_frame = frame
            write_and_show_image(frame, output_format, img_num, debug)
            img_num += 1
        elif not is_blurry:
            # Compare the current frame to the previous frame
            diff = compare_images(prev_frame, frame)
            # If the difference is within the allowed range, add it to the list of frames to consider
            if min_diff < diff < max_diff:
                frames_to_consider.append((frame, diff, blur_amnt))
            elif diff > max_diff and len(frames_to_consider) > 0:
                # find the least blurry frame in the list of frames to consider
                least_blurry = min(frames_to_consider, key=lambda x: x[2])
                # Write out the least blurry frame
                write_and_show_image(least_blurry[0], output_format, img_num, debug)
                img_num += 1

                # Clear the list of frames to consider
                frames_to_consider = []
                # And if the current frame's minimum difference is within the allowed range, add it to the list
                if min_diff < diff < max_diff:
                    frames_to_consider.append((frame, diff, blur_amnt))
            elif diff > max_diff and len(frames_to_consider) == 0:
                print("Warning! Frame difference between subsequent non-blurry frames is too high."
                      "Consider adjusting the program parameters or retaking a better video.")
                frames_to_consider.append((frame, diff, blur_amnt))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

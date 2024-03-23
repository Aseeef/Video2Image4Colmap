import argparse
import os
from tqdm import tqdm

import cv2 as cv
import numpy as np


class FrameData:
    def __init__(self, frame, diff, lv):
        self.frame = frame
        self.diff = diff
        self.lv = lv

    def __eq__(self, other):
        return self.diff == other.diff and self.lv == other.lv and np.equal(self.frame, other.frame).all()

    def __str__(self):
        return f"FrameData: diff={self.diff}, lv={self.lv}, frame_num={self.frame.get(cv.CAP_PROP_POS_FRAMES)}"

def compare_images(image1, image2):
    diff = cv.absdiff(image1, image2)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    mean, _ = cv.meanStdDev(diff)
    return mean[0]


def compare_images_with_orb(image1, image2) -> float:
    # Initialize ORB detector
    orb = cv.ORB.create(nfeatures=1000, scoreType=cv.ORB_FAST_SCORE, WTA_K=2, patchSize=31, edgeThreshold=31)

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Initialize a brute-force matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute similarity score (number of good matches)
    similarity = len(matches)

    # Normalize similarity score
    normalized_similarity = similarity / len(keypoints1)

    return normalized_similarity


# A function to quantify the amount of edges in an image
# The higher the value, the more edges there are which means the image is less blurry.
# However, the return value is not a direct measure of the amount of blur in the image
# and instead is a relative measure that depends on the scene.
def get_laplacian_variance(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image
    laplacian = cv.Laplacian(gray, cv.CV_64F)

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
    parser.add_argument('--output-format', type=str, default="output/output_%04d.png",
                        help='The output format for the images (default: output_%04d.png)')
    parser.add_argument('--max-diff', type=float, default=0.75,
                        help='The largest difference error allowed between frames (default: 0.85)')
    parser.add_argument('--min-diff', type=float, default=0.55,
                        help='The smallest difference error allowed between frames (default: 0.57)')
    parser.add_argument('--resize-scale', type=float, default=0.75,
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
    output_format = arguments.output_format
    max_diff = arguments.max_diff
    min_diff = arguments.min_diff
    resize_scale = arguments.resize_scale
    debug = arguments.debug

    capture = cv.VideoCapture(video_source)
    prev_frame = None
    frames_to_consider = []
    img_num = 0

    num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    res_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    res_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing video {video_source} with {num_frames} frames ({res_width}x{res_height})...")

    for _ in tqdm(range(num_frames)):
        success, frame = capture.read()
        # If the frame is not read, then we have reached the end of the video
        if not success:
            break
        # Resize the frame if needed
        # Note: downsizing can sharpen the image and make it less blurry
        if resize_scale != 1.0:
            frame = cv.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        # Show the current frame if debugging
        im_show_debug(frame, 'Current Frame', debug)
        # Determine if the frame is blurry and by how much
        lv = get_laplacian_variance(frame)
        # If this is the first non-blurry frame, write it out
        if prev_frame is None:
            prev_frame = frame
            write_and_show_image(frame, output_format, img_num, debug)
            img_num += 1
        else:
            # Compare the current frame to the previous frame
            diff = compare_images_with_orb(prev_frame, frame)
            frame = cv.putText(frame, "Difference: %f" % diff, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                               cv.LINE_AA)
            im_show_debug(frame, 'Difference', debug)

            # If the difference is within the allowed range, add it to the list of frames to consider
            if min_diff < diff < max_diff:
                frames_to_consider.append(FrameData(frame, diff, lv))
            elif diff < min_diff and len(frames_to_consider) > 0:
                # find the least blurry frame in the list of frames to consider
                least_blurry = min(frames_to_consider, key=lambda x: x.lv)
                least_blurry_index = frames_to_consider.index(least_blurry)
                # Write out the least blurry frame
                write_and_show_image(least_blurry.frame, output_format, img_num, debug)
                img_num += 1
                prev_frame = least_blurry.frame

                # Clear the list of frames including and before the current frame
                for i in range(least_blurry_index + 1):
                    frames_to_consider.pop(0)
                # Recalculate the difference for the current frame
                for f in frames_to_consider:
                    f.diff = compare_images_with_orb(prev_frame, f.frame)
                # Delete the frames where the difference is little
                frames_to_consider = [f for f in frames_to_consider if f.diff > max_diff]
                # And... add the current frame
                frames_to_consider.append(FrameData(frame, diff, lv))

    # Release the video capture object
    capture.release()
    # Close all OpenCV windows
    cv.destroyAllWindows()

    print("Done! Output a total of {img_num} images.")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

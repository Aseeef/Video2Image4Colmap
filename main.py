import argparse
import os
from typing import Union

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


def compare_images_with_orb(image1, image2) -> Union[float, None]:
    # Initialize ORB detector
    orb = cv.ORB.create(nfeatures=1000, scoreType=cv.ORB_FAST_SCORE, WTA_K=2, patchSize=31, edgeThreshold=31)

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # If no keypoints are found then probably the image is too blurry or captured a textureless surface
    if descriptors1 is None or descriptors2 is None:
        return None
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return None

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
    parser.add_argument('--output', type=str, default="output/output_%04d.png",
                        help='The output format for the images (default: output_%04d.png)')
    parser.add_argument('--max-siml', type=float, default=0.7,
                        help='The high similarity allowed between two subsequent outputs (default: 0.7)')
    parser.add_argument('--min-siml', type=float, default=0.45,
                        help='The lowest similarity allowed between two subsequent outputs (default: 0.45)')
    parser.add_argument('--resize-scale', type=float, default=1.0,
                        help='Whether to resize the output images (default: 1.0 (no resize))')
    parser.add_argument('--fps-scale', type=float, default=0.33,
                        help='The rescale ratio for the FPS (ex: 60fps * 0.33 = 20fps) (default: 0.33)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Show intermediate images for debugging')
    return parser.parse_args()


def write_and_show_image(image, output, img_num, debug):
    # create directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # write the image
    cv.imwrite(output % img_num, image)
    # show the image if debugging
    im_show_debug(image, 'Wrote Image', debug)


def im_show_debug(image, frame_name, debug, debug_img_text=None):
    if debug:
        if debug_img_text is not None:
            image = cv.putText(image, debug_img_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow(frame_name, image)
        cv.waitKey(0)


def validate_args(video_source, output, max_similarity, min_similarity, resize_scale):
    if min_similarity < 0.35:
        o = input("Warning: The min_similarity threshold is too low. If we continue like this, colmap will "
                  "likely have a lot of trouble reconstructing the scene. Continue anyways? (y/n)")
        if o.lower() != 'y':
            exit(1)
    if resize_scale > 1.0:
        o = input("Warning: The resize_scale is greater than 1.0. "
                  "This is likely a mistake as nothing good will come out of it. "
                  "Continue anyways? (y/n)")
        if o.lower() != 'y':
            exit(1)
    # check if file exists
    if not os.path.isfile(video_source):
        print(f"Error: The file {video_source} does not exist!")
        exit(1)
    # check write permissions
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if not os.access(os.path.dirname(output), os.W_OK):
        print(f"Error: The directory {os.path.dirname(output)} is not writable!")
        exit(1)
    # check if output dir is empty
    if len(os.listdir(os.path.dirname(output))) > 0:
        o = input(f"Warning: The directory {os.path.dirname(output)} is not empty. "
                  f"Files may be overwritten. Continue anyways? (y/n)")
        if o.lower() != 'y':
            exit(1)
        o = input(f"Would you like to clear the directory {os.path.dirname(output)}? (y/n)")
        if o.lower() == 'y':
            for f in os.listdir(os.path.dirname(output)):
                os.remove(os.path.join(os.path.dirname(output), f))
    # make sure the file name is not temp.mp4
    if output == "temp.mp4":
        print("Error: The output file name cannot be temp.mp4 because this name is reserved!")
        exit(1)


def main():
    arguments = get_program_params()

    video_source = arguments.arg1
    # video_source = "latest.mp4"
    output = arguments.output
    max_similarity = arguments.max_siml
    min_similarity = arguments.min_siml
    resize_scale = arguments.resize_scale
    # resize_scale = 0.5
    fps_scale = arguments.fps_scale
    debug = arguments.debug

    # Validate the arguments; exit if invalid args supplied
    validate_args(video_source, output, max_similarity, min_similarity, resize_scale)

    capture = cv.VideoCapture(video_source)

    prev_frame = None
    frames_to_consider = []
    frame_num = -1
    img_num = 0

    res_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    res_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    current_fps = capture.get(cv.CAP_PROP_FPS)
    num_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    print(f"Processing video {video_source} with {num_frames} frames ({res_width}x{res_height}) at {current_fps} FPS.")
    if fps_scale != 1.0 or resize_scale != 1.0:
        print(f"Resized frames by {resize_scale} and rescaling FPS by {fps_scale}.")
        print(f"New resolution: {int(res_width * resize_scale)}x{int(res_height * resize_scale)} "
              f"at {current_fps // int(1 / fps_scale)} FPS.")

    for _ in tqdm(range(num_frames), "Processing frames..."):
        success, frame = capture.read()
        # If the frame is not read, then we have reached the end of the video
        if not success:
            break

        frame_num += 1
        if frame_num % int(1 / fps_scale) != 0:
            continue

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
            write_and_show_image(frame, output, img_num, debug)
            img_num += 1
        else:
            # Compare the current frame to the previous frame
            similarity = compare_images_with_orb(prev_frame, frame)

            if similarity is None:
                print("Warning: No keypoints detected in the current frame. This indicates that the frame is either"
                      "too blurry or captured a textureless surface.")
                continue

            if similarity < min_similarity and len(frames_to_consider) == 0:
                print("Warning: The similarity between two subsequent frames was too low. "
                      "Either modify the min_similarity threshold or re-record the video "
                      "to get rid of this warning!")

            # If the difference is within the allowed range, add it to the list of frames to consider
            if min_similarity < similarity < max_similarity:
                frames_to_consider.append(FrameData(frame, similarity, lv))
            elif similarity < min_similarity and len(frames_to_consider) > 0:
                temp_sim = similarity
                # find the least blurry frame in the list of frames to consider
                least_blurry = min(frames_to_consider, key=lambda x: x.lv)
                least_blurry_index = frames_to_consider.index(least_blurry)
                # Write out the least blurry frame
                write_and_show_image(least_blurry.frame, output, img_num, debug)
                img_num += 1

                # Clear the list of frames including and before the current frame
                for i in range(least_blurry_index + 1):
                    frames_to_consider.pop(0)
                # Recalculate the difference for the current frame
                for f in frames_to_consider:
                    # Should never be None if I reasoned through this right
                    f.diff = compare_images_with_orb(prev_frame, f.frame)
                # Similarity between the written frame and the previous frame
                similarity = least_blurry.diff
                # Delete the frames where the difference is little
                frames_to_consider = [f for f in frames_to_consider if f.diff > max_similarity]
                # And... add the current frame
                frames_to_consider.append(FrameData(frame, temp_sim, lv))

                # set prev frame
                prev_frame = least_blurry.frame

            im_show_debug(frame, 'Difference', debug, "Difference: %f" % similarity)

    # Release the video capture object
    capture.release()
    # Close all OpenCV windows
    cv.destroyAllWindows()

    print(f"Done! Output a total of {img_num} images.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

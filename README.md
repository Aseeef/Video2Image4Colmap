## Basic Usage
`python main.py <video_input_path>`

## Additional Arguments
`--output`
The output format for the images (default: output/output_%04d.png).

`--max-siml`
The high similarity allowed between two subsequent frame outputs (default: 0.7). Together with `min-siml`, this decides how different the next frame must be to be extracted.

`--min-siml`
The lowest similarity allowed between two subsequent outputs (default: 0.45). Together with `max-siml`, this decides how different the next frame must be to be extracted.

`--resize-scale`
Whether to resize the output images (default: 1.0 (no resize)).

`--fps-scale`
The rescale ratio for the FPS (ex: 60fps * 0.33 = 20fps) (default: 0.33). A lower FPS decreases the runtime of the script but also increases the chances of missing potentially high quality frames.

`--debug`
Show intermediate images for debugging (default: false)

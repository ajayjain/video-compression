from PIL import Image
import numpy as np

def load_YUV420_video(path, dim):
    """Load a YUV420 video to a byte matrix and PIL images"""

    # Parameters for YUV 4:2:0 chroma subsampling format
    Y_bytes = dim[0] * dim[1]
    U_bytes = Y_bytes // 4
    V_bytes = Y_bytes // 4
    frame_bytes = Y_bytes + U_bytes + V_bytes

    # Read in video content as bytes
    video_content = None
    with open(path, 'rb') as video:
        video_content = video.read()

    # Count frames
    num_bytes = len(video_content)
    num_frames = num_bytes / (Y_bytes + U_bytes + V_bytes)

    byte_frames = []
    pil_frames = []

    # Read in each frame as a byte vector and as a PIL YCbCr image 
    for f in range(int(num_frames)):
        # Extract the Y, U, and V components of this frame
        bytes = video_content[f*frame_bytes : f*frame_bytes+Y_bytes+U_bytes+V_bytes]
        byte_vector = np.fromstring(bytes, dtype=np.uint8)
        byte_frames.append(byte_vector)

        y_bytes = video_content[f*frame_bytes : f*frame_bytes+Y_bytes]
        u_bytes = video_content[f*frame_bytes+Y_bytes : f*frame_bytes+Y_bytes+U_bytes]
        v_bytes = video_content[f*frame_bytes+Y_bytes+U_bytes : f*frame_bytes+Y_bytes+U_bytes+V_bytes]

        y = Image.frombytes('L', tuple(dim), y_bytes)
        u = Image.frombytes('L', (dim[0] // 2, dim[1] // 2), u_bytes).resize(dim)
        v = Image.frombytes('L', (dim[0] // 2, dim[1] // 2), v_bytes).resize(dim)

        frame = Image.merge('YCbCr', (y, u, v))
        pil_frames.append(frame)
    
    byte_matrix = np.stack(byte_frames, axis=1)
    
    return (byte_matrix, pil_frames)

def convert_pil_ycbcr_to_rgb(frames):
    rgb_frames = list(map(lambda frame: np.asarray(frame.convert('RGB')), frames))
    return rgb_frames

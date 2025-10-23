from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler, AutoencoderTiny
import numpy as np
import torch
import cv2
from PIL import Image
import argparse

# Choose device automatically: prefer CUDA, then MPS (Apple Silicon), else CPU
if torch.cuda.is_available():
    device = 'cuda'
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")

# Choose a safe dtype: use float16 on CUDA/MPS where supported, otherwise float32
if device == 'cuda' or device == 'mps':
    tdtype = torch.float16
else:
    tdtype = torch.float32

# load control net and stable diffusion v1-5
# load models with the chosen dtype/device
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=tdtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "lykon/dreamshaper-8-lcm", controlnet=controlnet, torch_dtype=tdtype, safety_checker=None
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# load small vae and move to device
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd")
try:
    pipe.vae = pipe.vae.to(device=device, dtype=tdtype)
except Exception:
    # some backends may not support dtype casting for the VAE; move only to device
    pipe.vae = pipe.vae.to(device=device)
#pipe.enable_model_cpu_offload()
pipe = pipe.to(device)
try:
    pipe.unet.to(memory_format=torch.channels_last)
except Exception:
    pass
# speed up diffusion process with faster scheduler and memory optimization

width = 640
height = 480
seed = 1231412
prompt = "cg, pixar, animation, 3d, character, design, concept, art, illustration, drawing, painting, digital"
negative_prompt = "realistic, portrait, photography, photo, human, face, people"


parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0, help='Camera device index')
parser.add_argument('--single', action='store_true', help='Run a single loop iteration and exit')
args = parser.parse_args()

# open camera (cross-platform)
cap = cv2.VideoCapture(args.camera)
# set camera resolution if supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# create a window and make it fullscreen on the second display
#cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Run the stream infinitely
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Warning: empty frame received from camera (index=', args.camera, '). Retrying...')
        # on single-run, exit with failure
        if args.single:
            break
        continue

    canny = cv2.Canny(frame, 100, 200)
    canny = canny[:, :, None]
    canny = np.concatenate([canny, canny, canny], axis=2)
    cv2.imshow("frame", np.array(frame))
    cv2.imshow("canny", np.array(canny))
    canny = Image.fromarray(canny)
    generator = torch.manual_seed(0)
    print("Generating image")
    x_output = pipe(prompt,
                    num_inference_steps=4, 
                    generator=generator, 
                    image=canny,
                    guidance_scale=1.2).images[0]
    cv2.imshow("Image",np.array(x_output) )
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if args.single:
        # allow the window to render then exit
        cv2.waitKey(500)
        break

cv2.destroyAllWindows()
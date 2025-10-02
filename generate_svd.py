import os
# os.environ['CUDA_VISIBLE_DEVICES']='5'
import argparse
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from torchvision.transforms import ToTensor
from diffusers.image_processor import VaeImageProcessor
from torchvision.transforms import functional as F
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--latent", type=str, default='')
    parser.add_argument("--output_name", type=str, default='')
    parser.add_argument("--ori_image_path", type=str, default='')
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)


    start_time = time.time()
    args = parser.parse_args()
    pipe = StableVideoDiffusionPipeline.from_pretrained(
    "your_path_to_stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to('cuda')

    if args.latent is not None:
        noisy_latents = torch.load(args.latent)
        noisy_latents = noisy_latents.permute(1,0,2,3).unsqueeze(0)
        l_latents = args.latent.split('/')[-1]
        start_index = l_latents[l_latents.find('_') + 1:l_latents.find('of')]
        start_index = int(start_index)
        num_inference_steps = int(args.latent[args.latent.find('of')+2:args.latent.find('.pt')])
    else:
        num_inference_steps = 25
        noisy_latents = None
        start_index = 0

    image = load_image(args.ori_image_path)
    generator = torch.manual_seed(args.seed)
    # os.makedirs(args.output_dir, exist_ok=True)
    output = pipe(image, decode_chunk_size=14, num_frames=args.num_frames, num_inference_steps = num_inference_steps, generator=generator, latents=noisy_latents, start_index=start_index).frames[0]
    # name = args.ori_image_path.split('/')[-1].split('.')[0] + '.mp4'
    # export_to_video(output, os.path.join(args.output_dir, name), fps=6)
    export_to_video(output, args.output_name, fps=6)

    end_time = time.time()
    print(end_time-start_time)

import os
# os.environ['CUDA_VISIBLE_DEVICES']='5'
import torch
import argparse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_file', type=str, default='')
 
    args = parser.parse_args()
    sd_pipeline = DiffusionPipeline.from_pretrained("your_path_to_stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=False, variant="fp16")
    sd_pipeline = sd_pipeline.to("cuda")
    image = sd_pipeline(prompt=args.prompt, height=args.height, width=args.width, num_inference_steps=50, seed=args.seed).images[0]

    # or use stable diffusion
    # sd_pipeline = StableDiffusionPipeline.from_pretrained('your_path_to_stable-diffusion-2-1', torch_dtype=torch.float16, use_safetensors=False, variant="fp16")
    # sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipeline.scheduler.config)
    # sd_pipeline = sd_pipeline.to("cuda")
    # image = sd_pipeline(args.prompt, args.height, args.width, num_inference_steps=50, seed=args.seed).images[0]
    image.save(args.output_file)
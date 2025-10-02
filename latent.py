import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import argparse
import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from PIL import Image
from torchvision.transforms import ToTensor
from diffusers.image_processor import VaeImageProcessor
from torchvision.transforms import functional as F

def concat_warp_start(image, num_frames, concat_path, device = 'cuda'):
    images = torch.Tensor([]).to(device)
    h, w = image.shape[2:]
    for i in range(num_frames):
        if i == 0:
            new_image = Image.open(f'{concat_path}/ori.png').resize((w, h)).convert('RGB')
            new_image = ToTensor()(new_image)
            new_image = new_image * 2.0 - 1.0
            new_image = new_image.unsqueeze(0).to(device)
        else:
            new_image = Image.open(f'{concat_path}/{i}_concat.png').resize((w, h)).convert('RGB')
            new_image = ToTensor()(new_image)
            new_image = new_image * 2.0 - 1.0
            new_image = new_image.unsqueeze(0).to(device)
        images = torch.cat([images, new_image])
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True) # the steps of adding noise
    parser.add_argument("--inference_steps", type=int, required=True)
    parser.add_argument("--warp_path", type=str, required=True) # warped images path
    parser.add_argument("--num_frames", type=int, required=True) # make sure this matches the frame number of video model
    args = parser.parse_args()

    model_path = 'your_path_to_CogVideoX-2b'
    dtype = torch.bfloat16
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe = pipe.to('cuda')
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    with torch.no_grad():
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor)
        warp_path = args.warp_path
        input_file = warp_path + '/ori.png'
        input_image = Image.open(input_file).convert('RGB')
        input_image = F.to_tensor(input_image).unsqueeze(0).to('cuda')
        input_images = concat_warp_start(input_image, args.num_frames, warp_path)
        input_images = image_processor.preprocess(input_images)
        input_images = input_images.permute(1,0,2,3).unsqueeze(0)
        input_images = input_images.to(dtype)
        latent_image = pipe.vae.encode(input_images).latent_dist.sample()
        latent_image = pipe.vae.config.scaling_factor * latent_image
        latent_images = latent_image.permute(0,2,1,3,4).to('cuda')
        index = args.index
        inference_steps = args.inference_steps
        noise = torch.randn_like(latent_images)
        pipe.scheduler.set_timesteps(num_inference_steps=inference_steps, device='cuda')
        timesteps = pipe.scheduler.timesteps[index].unsqueeze(0)
        noisy_latent = pipe.scheduler.add_noise(latent_images, noise, timesteps)
        noisy_latent = noisy_latent.to('cuda')
        torch.save(noisy_latent,f'{warp_path}/latents_{index}of{inference_steps}.pt')
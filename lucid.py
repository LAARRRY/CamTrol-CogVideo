import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import argparse
from PIL import Image
from camera import Warper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='')
    parser.add_argument("--prompt", type=str, default='') # used for inpainting
    # parser.add_argument("--pth", type=str, required=True) # directly load from camera coordinates
    parser.add_argument("--pcd_mode", type=str, default='zoom 1 14 out')
    parser.add_argument("--out_dir", type=str, default='')
    parser.add_argument("--H", type=int, default=576,)
    parser.add_argument("--W", type=int, default=1024,)
    args = parser.parse_args()

    H = args.H
    W = args.W
    input_image = Image.open(args.input_path)
    input_image = input_image.convert('RGB')
    prompt = args.prompt
    neg_prompt = 'out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature.'
    # pcd_mode = ['complex', '1', '14', args.pth]
    pcd_mode = args.pcd_mode.split(' ')
    num_steps = 25
    save_dir = args.out_dir
    os.makedirs(save_dir, exist_ok=True)
    input_image.save(os.path.join(save_dir, 'ori.png'))
    seed = 1
    warper = Warper(H, W)
    images = warper.generate_pcd(input_image, prompt, neg_prompt, pcd_mode, seed, num_steps, save_warps=True, save_dir = save_dir)


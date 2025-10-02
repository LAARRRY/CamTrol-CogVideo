# make sure to revise your_path_to_models to the paths of pretrained models

# generate a new image and save, or use your own
CUDA_VISIBLE_DEVICES=0 python image_generate_sdxl.py --prompt "" --output_file "" 

# generate warped images and save
# you will need a luciddreamer environment
CUDA_VISIBLE_DEVICES=0 python lucid.py --input_path "" --pcd_mode "" --out_dir "" --prompt ""

# turn warped images into latent
# different pretrained models have different schedules, here we use CogVideoX-2b as example
# use your own model's noise shedule, another example is in latent_svd.py
CUDA_VISIBLE_DEVICES=0 python latent.py --index 0 --warp_path ""

# generate video
# here use CogVideoX, example of using svd is in generate_svd.py
CUDA_VISIBLE_DEVICES=0 python cli_demo.py --output_path "" --latents "" --prompt ""

# if you have any questions you could also refer to another repository
# https://github.com/LAARRRY/CamTrol


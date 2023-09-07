import os
import copy
import argparse
import json

from distutils.util import strtobool

import numpy as np

import torch
import torch.nn as nn
from torch import optim

import torchvision
import cv2

from PIL import Image
from diffusers import AutoencoderKL

from unet import UNetModel
from character import *
from train import setup_logging, Diffusion, EMA


def save_single_images(images, path, args, **kwargs):
    #grid = torchvision.utils.make_grid(images, **kwargs)
    image = images.squeeze(0)
    print('images', image.shape)
    
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(image)
        #im = image.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(np.array(im))
    else:
        print('no latent')

    im.save(path)
    return im


def main():
    '''Main function'''
    
    path_str_type = lambda x: os.path.expanduser(str(x))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--img_size', type=int, default=(64, 64)) 
    parser.add_argument('--save_path', type=path_str_type, default='./datadisk/save_path/no_background inversed 64x64 ETL4,ETL5 epochs=1000')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--latent', type=strtobool, default=True)
    parser.add_argument('--interpolation', type=strtobool, default=False)
    parser.add_argument('--mix_rate', type=int, default=1)
    parser.add_argument('--stable_dif_path', type=path_str_type, default='~/datadisk/stable-diffusion-v1-5')
    parser.add_argument('--words_type', type=str, default="hiragana")
    
    args = parser.parse_args()

    setup_logging(args.save_path)
    
    with open(os.path.join(args.save_path, "writer2idx.json")) as f:
        writer2idx = json.load(f)
    num_classes = len(writer2idx)
    
    if args.words_type == "hiragana":
        words = hiraganas
    elif args.words_type == "katakana":
        words = katakanas
    else:
        raise Exception(f"unknown words_type: {args.words_type}")
    
    print('words', words)
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    if args.latent == True:
        unet = UNetModel(
            image_size = args.img_size, in_channels=4, model_channels=args.emb_dim, out_channels=4,
            num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
            num_heads=args.num_heads, num_classes=num_classes, context_dim=args.emb_dim, vocab_size=vocab_size,
            args=args
        ).to(args.device)
    else:
        unet = UNetModel(
            image_size = args.img_size, in_channels=3, model_channels=128, out_channels=3,
            num_res_blocks=1, attention_resolutions=(1, 2),
            num_heads=1, num_classes=num_classes, context_dim=128, vocab_size=vocab_size
        ).to(args.device)
    # unet = nn.DataParallel(unet, device_ids = [0,1,2,3,4]) #,5,6,7])
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    unet.load_state_dict(torch.load(os.path.join(args.save_path, "models", "ckpt.pt")))
    optimizer.load_state_dict(torch.load(os.path.join(args.save_path, "models", "optim.pt")))
    
    unet.eval()
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load(os.path.join(args.save_path, "models", "ema_ckpt.pt")))
    #ema_model = ema_model.to(args.device)
    ema_model.eval()
    
    if args.latent == True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    for w_name, w_idx in writer2idx.items():
        save_dir = os.path.join(args.save_path, "generated", "full", w_name)
        os.makedirs(save_dir, exist_ok=True)
        
        labels = torch.LongTensor([w_idx]).to(args.device)
        
        for x_text in words:
            ema_sampled_images = diffusion.sampling(
                ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args
            )
            sampled_ema = save_single_images(
                ema_sampled_images,
                os.path.join(save_dir, f"{char2code(x_text)}.png"),
                args
            )


if __name__ == "__main__":
    main()

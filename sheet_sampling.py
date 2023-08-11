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
from tqdm import tqdm
from diffusers import AutoencoderKL

from unet import UNetModel
from character import *
from train import setup_logging, Diffusion, EMA


def save_sheet(sheet, path, latent):
    n_kana = len(hiraganas)
    
    if len(sheet) != n_kana:
        raise Exception("illegal sheet")
    
    columns = []
    for i in range(0, n_kana, 5):
        items = sheet[i:(i + 5)]
        while len(items) < 5:
            items.append(torch.zeros(items[0].shape))
        columns.append(torch.cat(items, dim=1))
    columns.reverse()
    sheet = torch.cat(columns, dim=2)
    
    if latent == True:
        im = torchvision.transforms.ToPILImage()(sheet)
    else:
        ndarr = sheet.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
        
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
    parser.add_argument('--writers', type=str, nargs="*", default=["ETL4_5001", "ETL4_5002", "ETL5_6001", "ETL5_6002"])
    parser.add_argument('--interpolation', type=strtobool, default=False)
    parser.add_argument('--mix_rate', type=int, default=1)
    parser.add_argument('--stable_dif_path', type=path_str_type, default='~/datadisk/stable-diffusion-v1-5')
    parser.add_argument('--sheet_type', type=str, default="hiragana")
    
    args = parser.parse_args()

    setup_logging(args.save_path)
    
    with open(os.path.join(args.save_path, "writer2idx.json")) as f:
        writer2idx = json.load(f)
    num_classes = len(writer2idx)
    
    labels = torch.LongTensor([writer2idx[w] for w in args.writers]).to(args.device)
    
    if args.sheet_type == "hiragana":
        characters = hiraganas
    elif args.sheet_type == "katakana":
        characters = katakanas
    else:
        raise Exception(f"unknown sheet type: {args.sheet_type}")
    print('sheet type:', args.sheet_type)
    
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
    # ema_model = ema_model.to(args.device)
    ema_model.eval()
    
    if args.latent == True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None
    
    sheets = {w: [] for w in args.writers}
    for x_text in characters:
        ema_sampled_images = diffusion.sampling(
            ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args
        )
        for writer, image in zip(args.writers, ema_sampled_images):
            sheets[writer].append(image)
        
    for writer, sheet in sheets.items():
        save_sheet(
            sheet,
            os.path.join(args.save_path, 'generated', f"sheet_{args.sheet_type}_writer={writer}.jpg"),
            args.latent
        )
    

if __name__ == "__main__":
    main()

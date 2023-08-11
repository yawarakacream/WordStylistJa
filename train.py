import os
import copy
import argparse
import json

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import torchvision

from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL

from unet import UNetModel
from character import *


def setup_logging(save_path):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'generated'), exist_ok=True)


### Borrowed from GANwriting ###
def label_padding(labels, n_tokens):
    new_label_len = []
    ll = [char2idx[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + n_tokens
    ll = list(ll)
    #ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im


class EtlcdbDataset(Dataset):
    def __init__(self, etlcdb_path, etlcdb_names, etlcdb_process_type, transforms):
        self.transforms = transforms
        
        self.writer2idx = {} # {writer: writer_idx}
        self.writer_groups = {} # {etlcdb_names: writer[]}
        self.items = [] # (image_path, word, writer)
        
        writer_idx = 0
        for etlcdb_name in etlcdb_names:
            json_path = os.path.join(etlcdb_path, f"{etlcdb_name}.json")
            with open(json_path) as f:
                json_data = json.load(f)
            
            self.writer_groups[etlcdb_name] = []

            for item in json_data:
                relative_image_path = item["Path"] # ex) ETL4/5001/0x3042.png
                image_path = os.path.join(etlcdb_path, etlcdb_process_type, relative_image_path)
                
                character = item["Character"] # ex) "あ"
                
                # 対応していない文字はスルー
                if character not in char2idx:
                    continue
                
                word = [character]
                
                serial_sheet_number = int(item["Serial Sheet Number"]) # ex) 5001
                writer = f"{etlcdb_name}_{serial_sheet_number}"
                if writer not in self.writer2idx:
                    self.writer2idx[writer] = writer_idx
                    writer_idx += 1
                    self.writer_groups[etlcdb_name].append(writer)
                
                # データ拡張した構造
                if etlcdb_process_type.startswith("+"):
                    image_dir = image_path[:-len(".png")]
                    relative_image_paths = list(os.listdir(image_dir))
                    relative_image_paths.sort()
                    for relative_image_path in relative_image_paths:
                        image_path = os.path.join(image_dir, relative_image_path)
                        self.items.append((image_path, word, writer))
                
                # 素の構造
                else:
                    self.items.append((image_path, word, writer))
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        image_path, word, writer = self.items[idx]
        
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        
        word_embedding = label_padding(word, n_tokens) 
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()
        
        writer_idx = self.writer2idx[writer]
        
        return image, word_embedding, writer_idx

    
class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling(self, model, vae, n, x_text, labels, args, mix_rate=None, cfg_scale=3):
        model.eval()
        tensor_list = []
        # if mix_rate is not None:
        #     print('mix rate', mix_rate)
        with torch.no_grad():
            
            words = [x_text]*n
            for word in words:
                transcript = label_padding(word, n_tokens) #self.transform_text(transcript)
                word_embedding = np.array(transcript, dtype="int64")
                word_embedding = torch.from_numpy(word_embedding).long()#float()
                tensor_list.append(word_embedding)
            text_features = torch.stack(tensor_list)
            text_features = text_features.to(args.device)
            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                if cfg_scale > 0:
                    # uncond_predicted_noise = model(x, t, text_features, sid)
                    # predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    uncond_predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        if args.latent == True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
    
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
            
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
            
        return x


def train(
    diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, tests,
    num_classes, vocab_size, transforms, args, save_path
):
    checkpoint_epochs = set()
    checkpoint_epochs.add(0)
    tmp = 10 ** (len(str(args.epochs)) - 2)
    for i in range(10):
        if args.epochs < tmp * (i + 1) - 1:
            break
        checkpoint_epochs.add(tmp * (i + 1) - 1)
    checkpoint_epochs.add(args.epochs - 1)
    del tmp
    
    losses = []
    
    model.train()
    
    print('Training started....')
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)
        
        loss_sum = 0
        
        for i, (images, word, s_id) in enumerate(pbar):
            images = images.to(args.device)
            original_images = images
            text_features = word.to(args.device)
            
            s_id = s_id.to(args.device)
            
            if args.latent == True:
                images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images
            
            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            x_t, noise = diffusion.noise_images(images, t)
            
            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = model(
                x_t, original_images=original_images, timesteps=t, context=text_features, y=s_id, or_images=None
            )
            
            loss = mse_loss(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            
            pbar.set_postfix(MSE=loss.item())
            
            loss_sum += loss.item()
        
        losses.append(loss_sum / len(pbar))
        
        if epoch in checkpoint_epochs:
            words, writer_idxs = tests
            labels = torch.LongTensor(writer_idxs).to(args.device)
            n = len(labels)
            
            for x_text in words:
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args)
                sampled_ema = save_images(
                    ema_sampled_images,
                    os.path.join(save_path, 'images', f"{char2code(x_text)}_{epoch + 1}.jpg"),
                    args
                )
            
            torch.save(model.state_dict(), os.path.join(save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(save_path, "models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(save_path, "models", "optim.pt"))
    
    with open(os.path.join(save_path, "loss.log"), "w") as f:
        f.write("\n".join([str(l) for l in losses]))
    

def main():
    '''Main function'''
    path_str_type = lambda x: os.path.expanduser(str(x))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=(64, 64))
    parser.add_argument('--dataset', type=str, default='etlcdb')
    
    parser.add_argument("--etlcdb_path", type=path_str_type, default="~/datadisk/etlcdb_processed")
    parser.add_argument('--etlcdb_names', type=str, nargs="*", default=["ETL4", "ETL5"])
    parser.add_argument("--etlcdb_process_type", type=str, default="no_background 64x64")
    
    # UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=path_str_type, default='./datadisk/save_path')
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--stable_dif_path', type=path_str_type, default="~/datadisk/stable-diffusion-v1-5", help='path to stable diffusion')

    args = parser.parse_args()
    
    save_path = os.path.join(
        args.save_path,
        f"{args.etlcdb_process_type} {','.join(args.etlcdb_names)} epochs={args.epochs}"
    )
    
    # create save directories
    setup_logging(save_path)

    print('num of character classes', n_char_classes)
    print('num of tokens', n_tokens)
    print('character vocabulary size', vocab_size)
    
    if args.dataset == "etlcdb":
        # 謎
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        train_ds = EtlcdbDataset(args.etlcdb_path, args.etlcdb_names, args.etlcdb_process_type, transforms=transforms)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        n_style_classes = len(train_ds.writer2idx)
        
        with open(os.path.join(save_path, "writer2idx.json"), "w") as f:
            json.dump(train_ds.writer2idx, f, indent=2)
        
        tests = [[], []] # words, writers
        for etlcdb_name in args.etlcdb_names:
            if etlcdb_name == "ETL4":
                tests[0] += ["あ", "く", "さ"]
            elif etlcdb_name == "ETL5":
                tests[0] += ["ア", "コ", "ホ"]
                
            tests[1] += [train_ds.writer2idx[w] for w in train_ds.writer_groups[etlcdb_name][:16]]
    
    else:
        raise Exception("unknown dataset")
    
    with open(os.path.join(save_path, "char2idx.json"), "w") as f:
        json.dump(char2idx, f, indent=2)
    
    print(f"tests: {tests}")
    
    unet = UNetModel(
        image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels,
        num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1),
        num_heads=args.num_heads, num_classes=n_style_classes, context_dim=args.emb_dim, vocab_size=vocab_size,
        args=args, max_seq_len=OUTPUT_MAX_LEN
    ).to(args.device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    if args.latent:
        print('Latent is true - Working on latent space')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)

    else:
        print('Latent is false - Working on pixel space')
        vae = None

    train(
        diffusion, unet, ema, ema_model, vae, optimizer, mse_loss,
        train_loader, tests, n_style_classes, vocab_size, transforms, args, save_path
    )


if __name__ == "__main__":
    main()

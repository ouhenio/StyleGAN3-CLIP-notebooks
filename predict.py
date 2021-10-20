import os
import sys

sys.path.insert(0, 'CLIP')
sys.path.insert(0, 'stylegan3')
import tempfile
from pathlib import Path
from subprocess import call
import io
import pickle
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import clip
from tqdm import tqdm
from torchvision.transforms import Compose, Resize
from einops import rearrange
from subprocess import Popen, PIPE
import cog


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device('cuda:0')
        self.clip_model = CLIP()
        MODEL_NAME = {
            "FFHQ": "stylegan3-t-ffhqu-1024x1024.pkl",
            "MetFaces": "stylegan3-r-metfacesu-1024x1024.pkl",
            "AFHQv2": "stylegan3-t-afhqv2-512x512.pkl"
        }
        BASE_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"

        self.models = {}
        self.w_stdss = {}
        for key, value in MODEL_NAME.items():
            network_url = BASE_URL + MODEL_NAME[key]
            with open(fetch_model(network_url), 'rb') as fp:
                G = pickle.load(fp)['G_ema'].to(self.device)
                self.models[key] = G
            zs = torch.randn([10000, G.mapping.z_dim], device=self.device)
            self.w_stdss[key] = G.mapping(zs, None).std(0)

    @cog.input(
        "text",
        type=str,
        help="text for image generation",
    )
    @cog.input(
        "model_name",
        type=str,
        default='FFHQ',
        options=['FFHQ', 'MetFaces', 'AFHQv2'],
        help="choose model: FFHQ: human faces, MetFaces: human faces from works of art, AFHGv2: animal faces"
    )
    @cog.input(
        "output_type",
        type=str,
        default='mp4',
        options=['png', 'mp4'],
        help="choose output the final image or a video"
    )
    @cog.input(
        "steps",
        type=int,
        default=200,
        min=1,
        help="sampling steps, recommended to set <= 150 to avoid time out when choose video output"
    )
    @cog.input(
        "video_length",
        type=int,
        default=10,
        max=20,
        min=1,
        help="choose video length, valid if output is mp4"
    )
    @cog.input(
        "seed",
        type=int,
        default=2,
        help="set -1 for random seed"
    )
    def predict(self, text, model_name='FFHQ', steps=200, output_type='mp4', video_length=10, seed=2):
        if os.path.isdir('samples'):
            clean_folder('samples')
        os.makedirs(f'samples', exist_ok=True)

        G = self.models[model_name]
        w_stds = self.w_stdss[model_name]
        if not isinstance(seed, int):
            seed = 2
        if seed == -1:
            seed = np.random.randint(0, 9e9)
        target = self.clip_model.embed_text(text, self.device)

        # do run
        torch.manual_seed(seed)

        # Init
        # Sample 32 inits and choose the one closest to prompt

        with torch.no_grad():
            qs = []
            losses = []
            for _ in range(8):
                q = (G.mapping(torch.randn([4, G.mapping.z_dim], device=self.device), None,
                               truncation_psi=0.7) - G.mapping.w_avg) / w_stds
                images = G.synthesis(q * w_stds + G.mapping.w_avg)
                embeds = embed_image(images.add(1).div(2), self.clip_model)
                loss = spherical_dist_loss(embeds, target).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)
            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0).requires_grad_()

        # Sampling loop
        q_ema = q
        opt = torch.optim.AdamW([q], lr=0.08, betas=(0.0, 0.999))

        for i in range(steps):
            opt.zero_grad()
            w = q * w_stds
            image = G.synthesis(w + G.mapping.w_avg, noise_mode='const')
            embed = embed_image(image.add(1).div(2), self.clip_model)
            loss = spherical_dist_loss(embed, target).mean()
            loss.backward()
            opt.step()

            q_ema = q_ema * 0.9 + q * 0.1
            image = G.synthesis(q_ema * w_stds + G.mapping.w_avg, noise_mode='const')

            if i % 10 == 0:
                yield checkin(i, steps, loss, image)
            if output_type == 'mp4':
                pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0, 1))
                pil_image.save(f'samples/{i:04}.png')

        if output_type == 'png':
            out_path = Path(tempfile.mkdtemp()) / "out.png"
            TF.to_pil_image(image[0].add(1).div(2).clamp(0, 1)).save(str(out_path))
        else:
            out_path = Path(tempfile.mkdtemp()) / "out.mp4"
            make_video(str(out_path), video_length)

        return out_path


def make_video(out_path, video_length):
    frames = os.listdir('samples')
    frames = len(list(filter(lambda filename: filename.endswith(".png"), frames)))  # Get number of png generated

    init_frame = 1  # This is the frame where the video will start
    last_frame = frames

    min_fps = 10
    max_fps = 30

    total_frames = last_frame - init_frame

    frames = []
    tqdm.write('Generating video...')
    for i in range(init_frame, last_frame):
        filename = f"samples/{i:04}.png"
        frames.append(Image.open(filename))

    fps = np.clip(total_frames / video_length, min_fps, max_fps)

    p = Popen(
        ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264',
         '-r',
         str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', str(out_path)], stdin=PIPE)
    for im in tqdm(frames):
        im.save(p.stdin, 'PNG')
    p.stdin.close()
    p.wait()
    tqdm.write("The video is ready")


def checkin(i, steps, loss, image):
    tf = Compose([
        Resize(224),
        lambda x: torch.clamp((x + 1) / 2, min=0, max=1),
    ])
    tqdm.write(f"Image {i}/{steps} | Current loss: {loss}")
    img_path = Path(tempfile.mkdtemp()) / "progress.png"
    TF.to_pil_image(tf(image)[0]).save(str(img_path))
    return img_path


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def fetch_model(url_or_path):
    basename = os.path.basename(url_or_path)
    if os.path.exists(basename):
        return basename
    else:
        cmd = (
                "wget "
                + url_or_path
        )
        call(cmd, shell=True)
        return basename


def norm1(prompt):
    """Normalize to the unit sphere."""
    return prompt / prompt.square().sum(dim=-1, keepdim=True).sqrt()


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def embed_image(image, clip_model):
    n = image.shape[0]
    make_cutouts = MakeCutouts(224, 32, 0.5)
    cutouts = make_cutouts(image)
    embeds = clip_model.embed_cutout(cutouts)
    embeds = rearrange(embeds, '(cc n) c -> cc n c', n=n)
    return embeds


class CLIP(object):
    def __init__(self):
        clip_model = "ViT-B/32"
        self.model, _ = clip.load(clip_model)
        self.model = self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

    @torch.no_grad()
    def embed_text(self, prompt, device):
        """Normalized clip text embedding."""
        return norm1(self.model.encode_text(clip.tokenize(prompt).to(device)).float())

    def embed_cutout(self, image):
        """Normalized clip image embedding."""
        return norm1(self.model.encode_image(self.normalize(image)))


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

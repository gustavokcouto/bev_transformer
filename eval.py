from encoder import Encoder
from decoder import Decoder
from cvt import CrossViewTransformer
from efficientnet import EfficientNetExtractor
from nuscenes_dataset_generated import get_data
from vizualization import BaseViz, CLASSES
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
from matplotlib.pyplot import get_cmap
from expert_dataset import ExpertDataset
# from fvcore.nn import sigmoid_focal_loss
from pathlib import Path
from unet import GeneratorUNet


USE_UNET = True

class NuScenesViz(BaseViz):
    SEMANTICS = []


if __name__ == '__main__':
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_pixelwise = criterion_pixelwise.cuda()
    if USE_UNET:
        network = GeneratorUNet()
    else:
        image_height = 224
        image_width = 480
        backbone = EfficientNetExtractor(
            layer_names=['reduction_4'],
            image_height=image_height,
            image_width=image_width,
            model_name='efficientnet-b4'
        )
        cross_view = {
            'heads': 4,
            'dim_head': 32,
            'qkv_bias': True,
            'skip': True,
            'no_image_features': False,
            'image_height': image_height,
            'image_width': image_width
        }
        bev_embedding = {
            'sigma': 1.0,
            'bev_height': 200,
            'bev_width': 200,
            'h_meters': 100.0,
            'w_meters': 100.0,
            'offset': 0.0,
            'decoder_blocks': [128, 128, 64]
        }
        encoder_dim = 128

        encoder = Encoder(
            backbone=backbone,
            cross_view=cross_view,
            bev_embedding=bev_embedding,
            dim=encoder_dim,
            middle=[2],
            scale=1.0
        )

        decoder = Decoder(
            dim=encoder_dim,
            blocks=[128, 128, 64],
            residual=True,
            factor=2
        )

        network = CrossViewTransformer(
            encoder=encoder,
            decoder=decoder,
            dim_output=3,
            dim_last=64
        )

    batch_size = 8
    loader = torch.utils.data.DataLoader(
        ExpertDataset(
            'gail_experts',
            n_routes=1,
            n_eps=1,
            route_start=8,
            unet=USE_UNET
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network.to(device)
    network.train()

    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(exist_ok=True)
    eval_dir = Path('eval')
    eval_dir.mkdir(exist_ok=True)
    images = list()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0002, betas=(0.5, 0.999))
    n_epochs = [0, 10,20,24,30,40,49]
    img_verticals = [[] for _ in n_epochs]
    img_labels = []
    img_context_list = [[], [], [], []]
    img_ids = [221, 476, 527, 1280]
    for epoch_idx, epoch in enumerate(n_epochs):
        state_dict = torch.load(f'ckpt/ckpt_{epoch}.pth')
        network.load_state_dict(state_dict['network_state_dict'])
        network.eval()
        n_samples = 0
        last_samples = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                pred = network(batch)
                n_samples += batch['bev'].shape[0]
                for img_id in img_ids:
                    if img_id < last_samples or img_id >= n_samples:
                        continue
                    id = img_id - last_samples
                    bev = pred[id]
                    bev = bev.cpu().numpy()
                    bev = bev.transpose(1, 2, 0)
                    bev = (255 * bev).astype(np.uint8)
                    label = batch['bev'][id]
                    label = label.cpu().numpy()
                    label = label.transpose(1, 2, 0)
                    label = (255 * label).astype(np.uint8)
                    img_verticals[epoch_idx].append(bev)
                    if epoch_idx == 0:
                        img_labels.append(label)
                        for img_context_idx in range(len(img_context_list)):
                            img_context = batch['image'][id][img_context_idx]
                            img_context = img_context.cpu().numpy()
                            img_context = img_context.transpose(1, 2, 0)
                            img_context = 255 * img_context
                            img_context = img_context.astype(np.uint8)
                            img_context_list[img_context_idx].append(img_context)

                last_samples = n_samples

    for img_idx, img_v in enumerate(img_verticals):
        img_v = np.vstack(img_v)
        img_v = Image.fromarray(img_v)
        img_v = img_v.resize((144, 144 * 4), resample=Image.BILINEAR)
        img_v.save(eval_dir / f'bev_v_{n_epochs[img_idx]}.png')

    img_labels = np.vstack(img_labels)
    img_labels = Image.fromarray(img_labels)
    img_labels = img_labels.resize((144, 144 * 4), resample=Image.BILINEAR)
    img_labels.save(eval_dir / f'labels_v.png')

    for ctx_idx, img_context in enumerate(img_context_list):
        img_v = np.vstack(img_context)
        img_v = Image.fromarray(img_v)
        img_v = img_v.resize((144, 144 * 4), resample=Image.BILINEAR)
        img_v.save(eval_dir / f'ctx_{ctx_idx}.png')
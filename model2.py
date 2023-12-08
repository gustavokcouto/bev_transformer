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
from fvcore.nn import sigmoid_focal_loss
from pathlib import Path


class NuScenesViz(BaseViz):
    SEMANTICS = []


if __name__ == '__main__':
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_pixelwise = criterion_pixelwise.cuda()
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
            n_routes=8,
            n_eps=1,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    viz = NuScenesViz()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network.to(device)
    network.train()

    ckpt_dir = Path('ckpt')
    ckpt_dir.mkdir(exist_ok=True)
    eval_dir = Path('eval')
    eval_dir.mkdir(exist_ok=True)
    images = list()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0002, betas=(0.5, 0.999))
    n_epochs = 50
    for epoch in range(n_epochs):
        n_samples = 0
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = network(batch)
            loss = criterion_pixelwise(pred, batch['bev'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
            n_samples += batch['bev'].shape[0]
            epoch_loss += loss.item()
        print('epoch: ', epoch, ', loss: ', epoch_loss / n_samples)

        if epoch % 1 == 0:
            network.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    pred = network(batch)
                    break
                bev = pred[0]
                bev = bev.cpu().numpy()
                bev = bev.transpose(1, 2, 0)
                bev = (255 * bev).astype(np.uint8)
                label = batch['bev'][0]
                label = label.cpu().numpy()
                label = label.transpose(1, 2, 0)
                label = (255 * label).astype(np.uint8)
                x_img = Image.fromarray(bev)
                x_img.save(eval_dir / f'bev_{epoch}.png')
                x_img = Image.fromarray(label)
                x_img.save(eval_dir / f'label_{epoch}.png')
            ckpt_path = (ckpt_dir / f'ckpt_{epoch}.pth').as_posix()
            torch.save({
                'network_state_dict': network.state_dict()
            }, ckpt_path)
            network.train()

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


class NuScenesViz(BaseViz):
    SEMANTICS = []


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()

    for k, v in state_dict.items():
        tokens = k.split('.')

        if tokens[0] == prefix:
            tokens = tokens[1:]

        key = '.'.join(tokens)
        result[key] = v

    return result


if __name__ == '__main__':
    image_height = 224
    image_width = 480
    backbone = EfficientNetExtractor(
        layer_names=['reduction_2', 'reduction_4'],
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
        middle=[2, 2],
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
        dim_last=64,
        outputs={'bev': [0, 1]}
    )
    checkpoint = torch.load('logs/cvt_nuscenes_road_75k.ckpt')
    state_dict = remove_prefix(checkpoint['state_dict'], 'backbone')

    network.load_state_dict(state_dict)

    cfg_data = {
        'num_classes': 12,
        'version': 'v1.0-trainval',
        'dataset_dir': '/media/datasets/nuscenes',
        'labels_dir': '/media/datasets/cvt_labels_nuscenes',
        'cameras': [[0, 1, 2, 3, 4, 5]],
        'label_indices': None,
        'bev': {
            'h': 200,
            'w': 200,
            'h_meters': 100.0,
            'w_meters': 100.0,
            'offset': 0.0
        },
        'augment': 'none',
        'image': {
            'h': 224,
            'w': 480,
            'top_crop': 46
        }
    }
    SPLIT = 'val_qualitative_000'
    data_module = get_data(split=SPLIT, **cfg_data)

    SUBSAMPLE = 5
    dataset = torch.utils.data.ConcatDataset(data_module)
    dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    viz = NuScenesViz()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network.to(device)
    network.eval()

    images = list()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            for k, v in batch.items():
                print(k)
                print(v.shape)
            break
            # pred = network(batch)
            # bev = pred['bev'][0].sigmoid()
            # bev = bev.cpu().numpy()
            # bev = bev.transpose(1, 2, 0)
            # bev = bev.squeeze(2)
            # bev = (255 * bev).astype(np.uint8)
            # label_indices = [[0, 1]]
            # label = batch['bev']
            # label = [label[:, idx].max(1, keepdim=True).values for idx in label_indices]
            # label = torch.cat(label, 1)
            # label = label[0]
            # print(label.shape)
            # bev = (255 * get_cmap('inferno')(bev)[..., :3]).astype(np.uint8)
            # break
            # visualization = np.vstack(viz(batch=batch, pred=pred))
            # print(visualization.shape)
            # images.append(visualization)
    # x_img = Image.fromarray(bev)
    # x_img.save('bev.png')
    # x_img = Image.fromarray(bev)
    # x_img.save('label.png')
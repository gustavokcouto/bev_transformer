from encoder import Encoder
from decoder import Decoder
from cvt import CrossViewTransformer
from efficientnet import EfficientNetExtractor


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

    cvt = CrossViewTransformer(
        encoder=encoder,
        decoder=decoder,
        dim_last=64,
        outputs={'bev': [0, 1]}
    )

# modified from https://github.com/rohitrango/BC-regularized-GAIL/blob/master/a2c_ppo_acktr/algo/gail.py

from pathlib import Path
from typing import Union, Dict, Tuple, Any
from functools import partial
import time
import torch as th
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw


EXTRINSICS = [
        [[ 8.2272e-01, -5.6820e-01,  1.6596e-02, -9.1257e-01],
        [ 2.9975e-02,  1.4210e-02, -9.9945e-01,  1.4916e+00],
        [ 5.6766e-01,  8.2277e-01,  2.8723e-02, -1.2722e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 1.1086e-02, -9.9989e-01, -9.7436e-03,  9.0332e-03],
        [ 3.1933e-02,  1.0093e-02, -9.9944e-01,  1.4837e+00],
        [ 9.9943e-01,  1.0768e-02,  3.2042e-02, -1.6240e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-8.4308e-01, -5.3668e-01, -3.4468e-02,  9.9829e-01],
        [ 3.9957e-02,  1.4029e-03, -9.9920e-01,  1.4975e+00],
        [ 5.3630e-01, -8.4379e-01,  2.0261e-02, -1.2162e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-2.1208e-01, -3.1884e-01, -2.3248e-02,  5.4234e-01],
        [ 5.2442e-02,  -2.2342e-03, -7.43243e-01,  9.42334e+00],
        [ -2.5345e-01, 7.4324e-01,  -4.4324e-02, 2.4233e+00],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
        # [[ 8.2272e-01, -5.6820e-01,  1.6596e-02, -9.1257e-01],
        # [ 2.9975e-02,  1.4210e-02, -9.9945e-01,  1.4916e+00],
        # [ 5.6766e-01,  8.2277e-01,  2.8723e-02, -1.2722e+00],
        # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        # [[ 1.1086e-02, -9.9989e-01, -9.7436e-03,  9.0332e-03],
        # [ 3.1933e-02,  1.0093e-02, -9.9944e-01,  1.4837e+00],
        # [ 9.9943e-01,  1.0768e-02,  3.2042e-02, -1.6240e+00],
        # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        # [[-8.4308e-01, -5.3668e-01, -3.4468e-02,  9.9829e-01],
        # [ 3.9957e-02,  1.4029e-03, -9.9920e-01,  1.4975e+00],
        # [ 5.3630e-01, -8.4379e-01,  2.0261e-02, -1.2162e+00],
        # [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
]
INTRINSICS = [
        [[377.3588,   0.0000, 248.1723],
        [  0.0000, 377.3588,  89.2746],
        [  0.0000,   0.0000,   1.0000]],

        [[375.8439,   0.0000, 247.9764],
        [  0.0000, 375.8439,  94.9954],
        [  0.0000,   0.0000,   1.0000]],

        [[377.0246,   0.0000, 245.3366],
        [  0.0000, 377.0246,  89.5863],
        [  0.0000,   0.0000,   1.0000]],

        [[377.0246,   0.0000, 245.3366],
        [  0.0000, 377.0246,  89.5863],
        [  0.0000,   0.0000,   1.0000]],
        # [[377.3588,   0.0000, 248.1723],
        # [  0.0000, 377.3588,  89.2746],
        # [  0.0000,   0.0000,   1.0000]],

        # [[375.8439,   0.0000, 247.9764],
        # [  0.0000, 375.8439,  94.9954],
        # [  0.0000,   0.0000,   1.0000]],

        # [[377.0246,   0.0000, 245.3366],
        # [  0.0000, 377.0246,  89.5863],
        # [  0.0000,   0.0000,   1.0000]],
]



def traj_plotter_rgb(traj, bev_size, img_path=None):
    radius = 10
    color = (255, 255, 255)
    scale = 500
    point_idx = -1
    img = Image.fromarray(np.zeros((bev_size, bev_size, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    while (point_idx + 1) * 2 <= len(traj):
        if point_idx < 0:
            x = traj[1] * scale
            y = -1 * traj[0] * scale
        elif point_idx == 0:
            x = 0
            y = 0
        else:
            x = traj[point_idx*2 + 1] * scale
            y = -1 * traj[point_idx*2] * scale
        x += bev_size / 2
        y += bev_size - 40
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        point_idx += 1
    image_array = np.transpose(img, [2, 0, 1])
    image_tensor = th.as_tensor(image_array)
    return image_tensor


class ExpertDataset(th.utils.data.Dataset):
    def __init__(self, dataset_directory, n_routes=1, n_eps=1, route_start=0, ep_start=0, unet=False):
        self.dataset_path = Path(dataset_directory)
        self.length = 0
        self.get_idx = []
        self.trajs_states = []
        self.trajs_actions = []
        if unet:
            self.w_resize = 192
            self.h_resize = 192
            self.bev_resize = 192
        else:
            self.w_resize = 480
            self.h_resize = 224
            self.bev_resize = 200

        for route_idx in range(route_start, route_start + n_routes):
            for ep_idx in range(ep_start, ep_start + n_eps):
                route_path = self.dataset_path / ('route_%02d' % route_idx) / ('ep_%02d' % ep_idx)
                route_df = pd.read_json(route_path / 'episode.json')
                traj_length = route_df.shape[0]
                self.length += traj_length
                for step_idx in range(traj_length):
                    self.get_idx.append((route_idx, ep_idx, step_idx))
                    state_dict = {}
                    for state_key in route_df.columns:
                        state_dict[state_key] = route_df.iloc[step_idx][state_key]
                    self.trajs_states.append(state_dict)
                    self.trajs_actions.append(th.Tensor(route_df.iloc[step_idx]['actions']))

        self.trajs_actions = th.stack(self.trajs_actions)
        self.actual_obs = [None for _ in range(self.length)]

    def __len__(self):
        return self.length

    def process_image(self, image_path, birdview=False):
        image_array = Image.open(image_path)
        if birdview:
            image_array = image_array.resize((self.bev_resize, self.bev_resize), resample=Image.BILINEAR)
        else:
            image_array = image_array.resize((self.w_resize, self.h_resize), resample=Image.BILINEAR)
        image_array = np.transpose(image_array, [2, 0, 1])
        image_tensor = th.as_tensor(image_array.copy())
        image_tensor = image_tensor / 255.0
        return image_tensor

    def __getitem__(self, j):
        route_idx, ep_idx, step_idx = self.get_idx[j]
        if self.actual_obs[j] is None:
            # Load only the first time, images in uint8 are supposed to be light
            ep_dir = self.dataset_path / 'route_{:0>2d}/ep_{:0>2d}'.format(route_idx, ep_idx)
            masks_list = []
            for mask_index in range(1):
                mask_tensor = self.process_image(ep_dir / 'birdview_masks/{:0>4d}_{:0>2d}.png'.format(step_idx, mask_index), birdview=True)
                masks_list.append(mask_tensor)
            birdview = th.cat(masks_list)

            central_rgb = self.process_image(ep_dir / 'central_rgb/{:0>4d}.png'.format(step_idx))
            left_rgb = self.process_image(ep_dir / 'left_rgb/{:0>4d}.png'.format(step_idx))
            right_rgb = self.process_image(ep_dir / 'right_rgb/{:0>4d}.png'.format(step_idx))
            state_dict = self.trajs_states[j]
            traj_plot_rgb = traj_plotter_rgb(state_dict['traj'], self.bev_resize) / 255.0

            images = th.stack([left_rgb, central_rgb, right_rgb, traj_plot_rgb])
            extrinsics = th.Tensor(EXTRINSICS)
            intrinsics = th.Tensor(INTRINSICS)
            obs_dict = {
                'bev': birdview,
                'image': images,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics,
            }
            self.actual_obs[j] = obs_dict
        else:
            obs_dict = self.actual_obs[j]

        return obs_dict

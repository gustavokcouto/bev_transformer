import carla
import numpy as np
import pandas as pd
import tqdm

from PIL import Image
from pathlib import Path

from carla_gym.envs import LeaderboardEnv
from carla_gym.core.task_actor.scenario_actor.agents.constant_speed_agent import ConstantSpeedAgent
from carla_gym.utils.expert_noiser import ExpertNoiser
from rl_birdview_wrapper import RlBirdviewWrapper


reward_configs = {
    'hero': {
        'entry_point': 'reward.valeo_action:ValeoAction',
        'kwargs': {}
    }
}

terminal_configs = {
    'hero': {
        'entry_point': 'terminal.valeo_no_det_px:ValeoNoDetPx',
        'kwargs': {}
    }
}

env_configs = {
    'carla_map': 'Town01',
    'weather_group': 'dynamic_1.0',
    'routes_group': 'train'
}

obs_configs = {
    'hero': {
        'speed': {
            'module': 'actor_state.speed'
        },
        'control': {
            'module': 'actor_state.control'
        },
        'velocity': {
            'module': 'actor_state.velocity'
        },
        'birdview': {
            'module': 'birdview.chauffeurnet',
            'width_in_pixels': 192,
            'pixels_ev_to_bottom': 40,
            'pixels_per_meter': 5.0,
            'history_idx': [-16, -11, -6, -1],
            'scale_bbox': True,
            'scale_mask_col': 1.0
        },
        'route_plan': {
            'module': 'navigation.waypoint_plan',
            'steps': 20
        },
        'gnss': {
            'module': 'navigation.gnss'
        },    
        'central_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, 0.0, 1.3],
            'rotation': [0.0, 0.0, 0.0]
        },
        'left_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, -0.25, 1.3],
            'rotation': [0.0, 0.0, -45.0]
        },
        'right_rgb': {
            'module': 'camera.rgb',
            'fov': 90,
            'width': 256,
            'height': 144,
            'location': [1.2, 0.25, 1.3],
            'rotation': [0.0, 0.0, 45.0]
        }
    }
}

if __name__ == '__main__':
    env = LeaderboardEnv(obs_configs=obs_configs, reward_configs=reward_configs,
                         terminal_configs=terminal_configs, host="localhost", port=2000,
                         seed=2021, no_rendering=False, **env_configs)
    env = RlBirdviewWrapper(env)
    expert_file_dir = Path('gail_experts')
    expert_file_dir.mkdir(parents=True, exist_ok=True)
    # obs_metrics = ['control', 'vel_xy', 'linear_speed', 'vec', 'traj', 'cmd', 'command', 'state']
    for route_id in tqdm.tqdm(range(1)):
        env.set_task_idx(route_id)
        for ep_id in range(1):
            episode_dir = expert_file_dir / ('route_%02d' % route_id) / ('ep_%02d' % ep_id)
            (episode_dir / 'birdview_masks').mkdir(parents=True)
            (episode_dir / 'central_rgb').mkdir(parents=True)
            (episode_dir / 'left_rgb').mkdir(parents=True)
            (episode_dir / 'right_rgb').mkdir(parents=True)

            longitudinal_noiser = ExpertNoiser('Throttle', frequency=15, intensity=10, min_noise_time_amount=2.0)
            lateral_noiser = ExpertNoiser('Spike', frequency=25, intensity=4, min_noise_time_amount=0.5)

            obs = env.reset()
            basic_agent = ConstantSpeedAgent(env.env._ev_handler.ego_vehicles['hero'], None, 6.0)
            ep_dict = {}
            ep_dict['done'] = []
            ep_dict['actions'] = []
            ep_dict['state'] = []
            actions_ep = []
            i_step = 0
            c_route = False
            while not c_route:
                state_list = []
                ep_dict['done'].append(c_route)
                action = basic_agent.get_action()
                ep_dict['actions'].append([action[0], action[1]])
                birdview = obs['birdview']
                for i_mask in range(1):
                    birdview_mask = birdview[i_mask * 3: i_mask * 3 + 3]
                    birdview_mask = np.transpose(birdview_mask, [1, 2, 0]).astype(np.uint8)
                    Image.fromarray(birdview_mask).save(episode_dir / 'birdview_masks' / '{:0>4d}_{:0>2d}.png'.format(i_step, i_mask))

                central_rgb = obs['central_rgb']
                central_rgb = np.transpose(central_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(central_rgb).save(episode_dir / 'central_rgb' / '{:0>4d}.png'.format(i_step))

                left_rgb = obs['left_rgb']
                left_rgb = np.transpose(left_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(left_rgb).save(episode_dir / 'left_rgb' / '{:0>4d}.png'.format(i_step))

                right_rgb = obs['right_rgb']
                right_rgb = np.transpose(right_rgb, [1, 2, 0]).astype(np.uint8)
                Image.fromarray(right_rgb).save(episode_dir / 'right_rgb' / '{:0>4d}.png'.format(i_step))

                ep_dict['state'].append(obs['state'])

                obs, reward, done, info = env.step(action)
                c_route = info['route_completion']['is_route_completed']

                i_step += 1


            ep_df = pd.DataFrame(ep_dict)
            ep_df.to_json(episode_dir / 'episode.json')
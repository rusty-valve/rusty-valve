from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .h2_data import OUTPUT, load

def open_dataset(task, config, split='train'):
    path = OUTPUT / 'datasets' / task / split
    fps = float(load(path / 'meta/info.json')['fps'])
    return LeRobotDataset(f'primates-valve/v0_{task}_{split}',
        root=path, video_backend='pyav', download_videos=False,
        delta_timestamps={'action': [i / fps for i in config.action_delta_indices]})

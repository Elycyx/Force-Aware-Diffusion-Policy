#!/usr/bin/env python3
"""
将session_20251025_142256的HDF5格式数据转换为UMI数据集所需的zarr格式

HDF5数据格式:
- action: (T, 7) - [x, y, z, rx, ry, rz, gripper]
- state: (T, 7) - [x, y, z, rx, ry, rz, gripper]  
- image: (T, 480, 640, 3) - RGB图像
- force: (T, 6) - 力/力矩传感器数据
- timestamp*: 各种时间戳

目标zarr格式 (ReplayBuffer):
- data/camera0_rgb: RGB图像
- data/robot0_eef_pos: 末端执行器位置 (3,)
- data/robot0_eef_rot_axis_angle: 末端执行器旋转 (3,)
- data/robot0_gripper_width: 夹爪宽度 (1,)
- data/force: 末端执行器力/力矩 (6,) - [fx, fy, fz, mx, my, mz]
- data/action: 动作 (7,) - [x, y, z, rx, ry, rz, gripper]
- meta/episode_ends: 每个episode的结束索引
"""

import os
import sys
import pathlib
import h5py
import numpy as np
import zarr
from tqdm import tqdm
import cv2
import click
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import numcodecs

# 添加项目路径
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()


def _resize_single_image(args):
    """辅助函数：调整单张图像大小"""
    img, target_shape = args
    return cv2.resize(img, target_shape[::-1], interpolation=cv2.INTER_AREA)


def load_episode_hdf5(file_path, target_image_shape=(224, 224), num_workers=4):
    """
    从单个HDF5文件加载一个episode的数据
    
    Args:
        file_path: HDF5文件路径
        target_image_shape: 目标图像大小 (height, width)
        num_workers: 用于图像resize的线程数
    
    Returns:
        episode_data: 包含所有数据的字典
    """
    with h5py.File(file_path, 'r') as f:
        # 读取原始数据
        state = f['state'][:].astype(np.float32)  # (T, 7)
        action = f['action'][:].astype(np.float32)  # (T, 7)
        image = f['image'][:]  # (T, 480, 640, 3) uint8
        
        # 对action的rz（索引5）调整：第一个元素-pi/2，后面的元素-pi
        action[0, 5] -= np.pi / 2
        if len(action) > 1:
            action[1:, 5] -= np.pi
        # 将action的x和y坐标取反
        action[:, 0:2] *= -1
        
        # state使用当前action，action往后移一步
        # state[t]对应原始action[t]，action[t]对应原始action[t+1]
        state = action[:-1].copy()  # 去掉最后一个，因为最后一个state没有对应的下一步action
        action = action[1:].copy()  # 从第二个开始，表示从state[t]采取的动作
        image = image[:-1]  # image也去掉最后一帧，与state对齐
        
        n_steps = state.shape[0]  # 现在比原始数据少1个时间步
        
        # 读取force数据（如果存在）
        if 'force' in f:
            force = f['force'][:].astype(np.float32)  # (T, 6)
            force = force[:-1]  # force也去掉最后一个，与state对齐
            # 确保force长度与state一致
            if force.shape[0] != n_steps:
                print(f"警告: {pathlib.Path(file_path).name} 中force数据长度({force.shape[0]})与state({n_steps})不一致，截断或填充")
                if force.shape[0] > n_steps:
                    force = force[:n_steps]
                else:
                    # 填充零数组
                    padding = np.zeros((n_steps - force.shape[0], 6), dtype=np.float32)
                    force = np.concatenate([force, padding], axis=0)
        else:
            # 如果force数据不存在，创建零数组
            force = np.zeros((n_steps, 6), dtype=np.float32)
            print(f"警告: {pathlib.Path(file_path).name} 中未找到force数据，使用零数组填充")
        
        # 使用多线程批量调整图像大小
        if num_workers > 1 and n_steps > 10:
            # 对于大量图像，使用多线程并行处理
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                resize_args = [(image[i], target_image_shape) for i in range(n_steps)]
                resized_images = list(executor.map(_resize_single_image, resize_args))
            resized_images = np.stack(resized_images, axis=0)  # (T, H, W, 3)
        else:
            # 对于少量图像，直接处理（避免线程开销）
            resized_images = np.zeros((n_steps, target_image_shape[0], target_image_shape[1], 3), dtype=np.uint8)
            for i in range(n_steps):
                resized_images[i] = cv2.resize(image[i], target_image_shape[::-1], 
                                             interpolation=cv2.INTER_AREA)
        
        # 准备episode数据
        episode_data = {
            'camera0_rgb': resized_images,  # (T, H, W, 3)
            'robot0_eef_pos': state[:, :3],  # (T, 3)
            'robot0_eef_rot_axis_angle': state[:, 3:6],  # (T, 3)
            'robot0_gripper_width': state[:, 6:7],  # (T, 1)
            'force': force,  # (T, 6) - [fx, fy, fz, mx, my, mz]
            'action': action,  # (T, 7)
        }
        
        return episode_data


def _load_episode_wrapper(args):
    """包装函数，用于多进程处理"""
    file_path, target_image_shape, num_workers = args
    try:
        return load_episode_hdf5(file_path, target_image_shape, num_workers)
    except Exception as e:
        return (file_path, e)  # 返回错误信息


def convert_session_to_zarr(
    session_dir,
    output_path,
    target_image_shape=(224, 224),
    max_episodes=None,
    num_workers=4,
    use_multiprocessing=False,
    fast_save=False
):
    """
    将整个session目录下的HDF5文件转换为单个zarr文件
    
    Args:
        session_dir: session目录路径
        output_path: 输出zarr文件路径 (.zarr.zip)
        target_image_shape: 目标图像大小 (height, width)
        max_episodes: 最大episode数量，None表示转换全部
        num_workers: 用于图像resize和episode处理的线程/进程数
        use_multiprocessing: 是否使用多进程处理episodes（更快但占用更多内存）
        fast_save: 是否使用快速保存模式（更快但文件更大）
    """
    session_dir = pathlib.Path(session_dir)
    output_path = pathlib.Path(output_path)
    
    # 查找所有episode文件
    episode_files = sorted(session_dir.glob('episode*.hdf5'), 
                          key=lambda x: int(x.stem.replace('episode', '')))
    
    if max_episodes is not None:
        episode_files = episode_files[:max_episodes]
    
    print(f"找到 {len(episode_files)} 个episode文件")
    
    # 创建ReplayBuffer
    replay_buffer = ReplayBuffer.create_empty_numpy()
    
    # 并行处理episodes
    if use_multiprocessing and len(episode_files) > 1:
        # 使用多进程并行处理多个episodes
        print(f"使用多进程并行处理（{num_workers}个进程）...")
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                load_args = [(str(f), target_image_shape, min(2, num_workers)) for f in episode_files]
                results = list(tqdm(
                    executor.map(_load_episode_wrapper, load_args),
                    total=len(episode_files),
                    desc="转换episodes"
                ))
            
            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Exception):
                    # 处理错误
                    print(f"警告: 无法加载 {episode_files[i].name}: {result[1]}")
                else:
                    replay_buffer.add_episode(result)
        except Exception as e:
            print(f"警告: 多进程处理失败，回退到顺序处理: {e}")
            # 回退到顺序处理
            use_multiprocessing = False
    
    if not use_multiprocessing:
        # 顺序处理（但每个episode内部使用多线程）
        for episode_file in tqdm(episode_files, desc="转换episodes"):
            try:
                episode_data = load_episode_hdf5(
                    episode_file, 
                    target_image_shape=target_image_shape,
                    num_workers=num_workers
                )
                replay_buffer.add_episode(episode_data)
            except Exception as e:
                print(f"警告: 无法加载 {episode_file.name}: {e}")
                continue
    
    print(f"\n总共加载了 {replay_buffer.n_episodes} 个episodes")
    print(f"总步数: {replay_buffer.n_steps}")
    
    # 打印数据统计信息
    print("\n数据统计:")
    for key in replay_buffer.data.keys():
        data = replay_buffer.data[key]
        print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
        if data.dtype in [np.float32, np.float64]:
            print(f"    范围: [{data.min():.3f}, {data.max():.3f}]")
    
    # 保存到zarr文件
    print(f"\n保存到: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据fast_save选项选择压缩设置
    if fast_save:
        # 快速保存：使用最快的压缩（低压缩级别）
        fast_compressor = numcodecs.Blosc(
            cname='lz4', 
            clevel=1,  # 最低压缩级别，最快
            shuffle=numcodecs.Blosc.NOSHUFFLE
        )
        compressors = {
            'camera0_rgb': fast_compressor,
            'default': fast_compressor
        }
        print("使用快速保存模式（最低压缩，最快但文件最大）...")
    else:
        # 默认保存：使用较快的lz4压缩（平衡速度和文件大小）
        default_compressor = numcodecs.Blosc(
            cname='lz4',
            clevel=3,  # 中等压缩级别，较快且文件大小合理
            shuffle=numcodecs.Blosc.NOSHUFFLE
        )
        compressors = {
            'camera0_rgb': default_compressor,
            'default': default_compressor
        }
        print("使用默认保存模式（中等压缩，平衡速度和文件大小）...")
    
    # 保存到zarr
    with tqdm(total=1, desc="保存zarr文件", unit="文件") as pbar:
        if output_path.suffix == '.zip' or str(output_path).endswith('.zarr.zip'):
            with zarr.ZipStore(str(output_path), mode='w') as zip_store:
                replay_buffer.save_to_store(
                    store=zip_store,
                    chunks={
                        'camera0_rgb': (1, target_image_shape[0], target_image_shape[1], 3),
                        'robot0_eef_pos': (100, 3),
                        'robot0_eef_rot_axis_angle': (100, 3),
                        'robot0_gripper_width': (100, 1),
                        'force': (100, 6),
                        'action': (100, 7),
                    },
                    compressors=compressors
                )
        else:
            with zarr.DirectoryStore(str(output_path)) as store:
                replay_buffer.save_to_store(
                    store=store,
                    chunks={
                        'camera0_rgb': (1, target_image_shape[0], target_image_shape[1], 3),
                        'robot0_eef_pos': (100, 3),
                        'robot0_eef_rot_axis_angle': (100, 3),
                        'robot0_gripper_width': (100, 1),
                        'force': (100, 6),
                        'action': (100, 7),
                    },
                    compressors=compressors
                )
        pbar.update(1)
    
    print("转换完成!")
    return replay_buffer


@click.command()
@click.option('--input', '-i', required=True, 
              help='输入session目录路径')
@click.option('--output', '-o', default=None,
              help='输出zarr文件路径 (默认: <input>/dataset.zarr.zip)')
@click.option('--image-size', '-s', default='224x224',
              help='目标图像大小，格式: WIDTHxHEIGHT (默认: 224x224)')
@click.option('--max-episodes', '-n', default=None, type=int,
              help='最大episode数量 (默认: 全部)')
@click.option('--num-workers', '-w', default=4, type=int,
              help='并行处理的线程/进程数 (默认: 4)')
@click.option('--use-multiprocessing', is_flag=True,
              help='使用多进程并行处理episodes（更快但占用更多内存）')
@click.option('--fast-save', is_flag=True,
              help='使用快速保存模式（更快但文件更大，使用低压缩级别）')
def main(input, output, image_size, max_episodes, num_workers, use_multiprocessing, fast_save):
    """
    将HDF5格式的session数据转换为UMI数据集所需的zarr格式
    
    示例:
        python convert_session_to_zarr.py -i data/session_20251025_142256
        python convert_session_to_zarr.py -i data/session_20251025_142256 -o my_dataset.zarr.zip
        python convert_session_to_zarr.py -i data/session_20251025_142256 -s 256x256 -n 10
        python convert_session_to_zarr.py -i data/session_20251025_142256 -w 8 --use-multiprocessing
        python convert_session_to_zarr.py -i data/session_20251025_142256 --fast-save
    """
    # 解析图像大小
    width, height = map(int, image_size.split('x'))
    target_image_shape = (height, width)
    
    # 设置输入输出路径
    input_path = pathlib.Path(input)
    if output is None:
        output_path = input_path / 'dataset.zarr.zip'
    else:
        output_path = pathlib.Path(output)
    
    # 检查输入路径
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 如果输出文件已存在，询问是否覆盖
    if output_path.exists():
        click.confirm(f'输出文件已存在: {output_path}\n是否覆盖?', abort=True)
    
    # 执行转换
    convert_session_to_zarr(
        session_dir=input_path,
        output_path=output_path,
        target_image_shape=target_image_shape,
        max_episodes=max_episodes,
        num_workers=num_workers,
        use_multiprocessing=use_multiprocessing,
        fast_save=fast_save
    )


if __name__ == '__main__':
    main()


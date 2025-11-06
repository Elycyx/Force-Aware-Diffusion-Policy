#!/usr/bin/env python3
"""
查看和检查zarr数据集文件的内容

用法:
    python inspect_zarr.py <zarr_file_path>
    python inspect_zarr.py <zarr_file_path> --detailed
    python inspect_zarr.py <zarr_file_path> --episode 0
    python inspect_zarr.py <zarr_file_path> --visualize
"""

import sys
import pathlib
import numpy as np
import zarr
import click
from typing import Optional

# 添加项目路径
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def inspect_zarr_structure(zarr_path):
    """显示zarr文件的基本结构"""
    print_separator()
    print(f"检查文件: {zarr_path}")
    print_separator()
    
    if str(zarr_path).endswith('.zip') or str(zarr_path).endswith('.zarr.zip'):
        store = zarr.ZipStore(str(zarr_path), mode='r')
    else:
        store = zarr.DirectoryStore(str(zarr_path))
    
    root = zarr.group(store)
    
    # 显示完整树形结构
    print("\n文件结构:")
    print(root.tree())
    
    return root, store


def show_basic_info(root):
    """显示基本信息"""
    print_separator('-')
    print("基本信息:")
    print_separator('-')
    
    # Episode信息
    if 'meta' in root and 'episode_ends' in root['meta']:
        episode_ends = root['meta']['episode_ends'][:]
        n_episodes = len(episode_ends)
        n_steps = episode_ends[-1] if n_episodes > 0 else 0
        
        print(f"Episodes数量: {n_episodes}")
        print(f"总步数: {n_steps}")
        
        if n_episodes > 0:
            episode_starts = np.concatenate([[0], episode_ends[:-1]])
            episode_lengths = episode_ends - episode_starts
            print(f"Episode长度统计:")
            print(f"  最小: {episode_lengths.min()}")
            print(f"  最大: {episode_lengths.max()}")
            print(f"  平均: {episode_lengths.mean():.1f}")
            print(f"  中位数: {np.median(episode_lengths):.1f}")
    
    # 数据键
    if 'data' in root:
        print(f"\n数据键: {list(root['data'].keys())}")


def show_data_details(root):
    """显示详细的数据信息"""
    print_separator('-')
    print("数据详情:")
    print_separator('-')
    
    if 'data' not in root:
        print("没有找到data组")
        return
    
    for key in sorted(root['data'].keys()):
        data = root['data'][key]
        print(f"\n[{key}]")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  块大小: {data.chunks}")
        print(f"  压缩器: {data.compressor}")
        
        # 数值统计（仅针对数值类型）
        if data.dtype in [np.float32, np.float64, np.int32, np.int64]:
            if data.size > 0:
                sample = data[:min(1000, data.shape[0])]  # 采样前1000步
                print(f"  数值范围: [{np.min(sample):.4f}, {np.max(sample):.4f}]")
                print(f"  均值: {np.mean(sample):.4f}")
                print(f"  标准差: {np.std(sample):.4f}")
        elif data.dtype == np.uint8:
            print(f"  数值范围: [0, 255] (图像数据)")


def show_episode_data(root, episode_idx):
    """显示特定episode的数据"""
    print_separator('-')
    print(f"Episode {episode_idx} 详情:")
    print_separator('-')
    
    if 'meta' not in root or 'episode_ends' in root['meta']:
        episode_ends = root['meta']['episode_ends'][:]
        n_episodes = len(episode_ends)
        
        if episode_idx >= n_episodes:
            print(f"错误: Episode {episode_idx} 不存在 (共{n_episodes}个episodes)")
            return
        
        # 计算episode范围
        start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
        end_idx = episode_ends[episode_idx]
        length = end_idx - start_idx
        
        print(f"索引范围: [{start_idx}, {end_idx})")
        print(f"长度: {length} 步")
        
        # 显示各个数据的片段
        if 'data' in root:
            for key in sorted(root['data'].keys()):
                data = root['data'][key]
                episode_data = data[start_idx:end_idx]
                print(f"\n[{key}]")
                print(f"  形状: {episode_data.shape}")
                
                if data.dtype in [np.float32, np.float64]:
                    print(f"  范围: [{np.min(episode_data):.4f}, {np.max(episode_data):.4f}]")
                    print(f"  前3步数据:")
                    for i in range(min(3, episode_data.shape[0])):
                        if len(episode_data.shape) == 2:
                            print(f"    步{i}: {episode_data[i]}")
                        else:
                            print(f"    步{i}: shape={episode_data[i].shape}")


def visualize_episodes(root, max_episodes=10):
    """可视化episodes的长度分布"""
    print_separator('-')
    print("Episode长度可视化:")
    print_separator('-')
    
    if 'meta' not in root or 'episode_ends' not in root['meta']:
        print("无法找到episode信息")
        return
    
    episode_ends = root['meta']['episode_ends'][:]
    n_episodes = len(episode_ends)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts
    
    # 显示前N个episodes的长度
    show_count = min(max_episodes, n_episodes)
    print(f"\n前{show_count}个episodes的长度:")
    
    max_length = episode_lengths[:show_count].max()
    for i in range(show_count):
        length = episode_lengths[i]
        bar_length = int(50 * length / max_length)
        bar = '█' * bar_length
        print(f"  Episode {i:3d}: {bar} {length}")
    
    # 长度分布直方图
    if n_episodes > 1:
        print(f"\n长度分布 (所有{n_episodes}个episodes):")
        bins = np.linspace(episode_lengths.min(), episode_lengths.max(), 11)
        hist, _ = np.histogram(episode_lengths, bins=bins)
        max_count = hist.max()
        
        for i in range(len(hist)):
            count = hist[i]
            bar_length = int(40 * count / max_count) if max_count > 0 else 0
            bar = '█' * bar_length
            print(f"  {bins[i]:6.0f}-{bins[i+1]:6.0f}: {bar} ({count})")


def visualize_image_sample(root, step_idx=0):
    """显示图像样本（需要matplotlib）"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装matplotlib来显示图像: pip install matplotlib")
        return
    
    print_separator('-')
    print(f"显示第{step_idx}步的图像:")
    print_separator('-')
    
    if 'data' not in root:
        print("没有找到data组")
        return
    
    # 查找RGB图像数据
    rgb_keys = [k for k in root['data'].keys() if 'rgb' in k.lower() or 'image' in k.lower()]
    
    if not rgb_keys:
        print("没有找到图像数据")
        return
    
    n_images = len(rgb_keys)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    if n_images == 1:
        axes = [axes]
    
    for idx, key in enumerate(rgb_keys):
        img_data = root['data'][key]
        if step_idx >= img_data.shape[0]:
            print(f"警告: 步数{step_idx}超出范围 (最大{img_data.shape[0]-1})")
            return
        
        img = img_data[step_idx]
        axes[idx].imshow(img)
        axes[idx].set_title(f"{key}\n形状: {img.shape}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('zarr_image_sample.png', dpi=150, bbox_inches='tight')
    print(f"图像已保存到: zarr_image_sample.png")
    plt.close()


@click.command()
@click.argument('zarr_path', type=click.Path(exists=True))
@click.option('--detailed', '-d', is_flag=True, 
              help='显示详细的数据统计信息')
@click.option('--episode', '-e', type=int, default=None,
              help='显示特定episode的详细信息')
@click.option('--visualize', '-v', is_flag=True,
              help='可视化episode长度分布')
@click.option('--show-image', '-i', type=int, default=None,
              help='显示特定步数的图像（需要matplotlib）')
@click.option('--max-episodes', '-m', type=int, default=10,
              help='可视化时显示的最大episode数量')
def main(zarr_path, detailed, episode, visualize, show_image, max_episodes):
    """
    查看和检查zarr数据集文件的内容
    
    示例:
        python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip
        python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --detailed
        python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --episode 0
        python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --visualize
        python inspect_zarr.py data/session_20251025_142256/dataset.zarr.zip --show-image 0
    """
    zarr_path = pathlib.Path(zarr_path)
    
    try:
        # 显示基本结构
        root, store = inspect_zarr_structure(zarr_path)
        
        # 显示基本信息
        show_basic_info(root)
        
        # 详细信息
        if detailed:
            show_data_details(root)
        
        # 特定episode
        if episode is not None:
            show_episode_data(root, episode)
        
        # 可视化
        if visualize:
            visualize_episodes(root, max_episodes)
        
        # 显示图像
        if show_image is not None:
            visualize_image_sample(root, show_image)
        
        # 关闭store
        if hasattr(store, 'close'):
            store.close()
        
        print_separator()
        print("检查完成!")
        print_separator()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


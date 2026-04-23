import os
import sys
import glob
import numpy as np
import argparse

# 确保路径配置正确
CGN_REPO_ROOT = r"/home/zyp/pan1/contact_graspnet_pytorch"
if CGN_REPO_ROOT not in sys.path:
    sys.path.append(CGN_REPO_ROOT)
CGN_SRC_DIR = os.path.join(CGN_REPO_ROOT, "contact_graspnet_pytorch")
if CGN_SRC_DIR not in sys.path:
    sys.path.append(CGN_SRC_DIR)

from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps

def show_saved_vis(npz_path, show_all=False):
    print(f"👉 正在加载离线数据: {os.path.basename(npz_path)}")
    data = np.load(npz_path)
    
    best_idx = int(data['best_idx'])
    obj_id = 1
    
    if show_all:
        # 显示所有抓取框 (一团乱麻模式)
        vis_grasps = {obj_id: data['grasps']}
        vis_scores = {obj_id: data['scores']}
        vis_openings = {obj_id: data['openings']}
        print(f"⭐ 当前显示: 【所有】候选抓取 (最高分索引: {best_idx})")
    else:
        # 🌟 核心改进：利用切片 [best_idx : best_idx+1] 
        # 仅把那 1 个最高分的抓取框丢给渲染器，保持维度正确
        vis_grasps = {obj_id: data['grasps'][best_idx : best_idx+1]}
        vis_scores = {obj_id: data['scores'][best_idx : best_idx+1]}
        vis_openings = {obj_id: data['openings'][best_idx : best_idx+1]}
        print(f"🎯 当前显示: 【仅最高分】抓取 (得分: {data['scores'][best_idx]:.4f})")
    
    print("👀 请在弹出的 Open3D 窗口中查看。看完后关闭窗口，会自动加载下一个...")
    
    # 弹出 Open3D 窗口
    visualize_grasps(
        data['pc_full'], 
        vis_grasps, 
        vis_scores, 
        plot_opencv_cam=True, 
        pc_colors=data['pc_colors'],
        gripper_openings=vis_openings
    )

def batch_visualize(directory, show_all=False):
    # 搜索目录下所有匹配 trial_*_cam*_cgn_vis.npz 格式的文件
    search_pattern = os.path.join(directory, "trial_*_cam*_cgn_vis.npz")
    file_list = glob.glob(search_pattern)
    
    # 按照文件名排序，确保按照顺序播放
    file_list.sort()

    if not file_list:
        print(f"❌ 在 {directory} 目录下没有找到任何匹配的 .npz 可视化文件！")
        return

    print(f"🔍 共找到 {len(file_list)} 个可视化文件，准备依次播放...")
    print("=====================================================")
    
    for i, npz_file in enumerate(file_list):
        print(f"\n▶ 进度: [{i+1}/{len(file_list)}]")
        show_saved_vis(npz_file, show_all)

    print("\n🎉 所有文件已查看完毕！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线批量查看 CGN 抓取结果")
    DEFAULT_DIR = "/home/zyp/Desktop/eval_results100"
    
    parser.add_argument("--dir", type=str, default=DEFAULT_DIR, 
                        help="存放 .npz 文件的目录路径")
    # 🌟 新增开关：默认不带它就是只看最高分，带上它就看全部
    parser.add_argument("--all", action="store_true", 
                        help="显示所有候选抓取（默认只显示唯一最高分）")
    args = parser.parse_args()
    
    batch_visualize(args.dir, args.all)
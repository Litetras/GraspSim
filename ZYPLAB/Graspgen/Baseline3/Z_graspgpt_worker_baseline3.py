# Z_graspgpt_worker_baseline3.py
import open3d as o3d
import argparse
import os
import sys
import numpy as np
import torch

# ===================== 🌟 核心修复：抵抗 Isaac Sim 的环境变量污染 =====================
os.environ['HF_HOME'] = '/home/zyp/.cache/huggingface'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  
# ====================================================================================

BASE_DIR = "/home/zyp/pan1/GraspGPT_public" 
GCNGRASP_DIR = "/home/zyp/pan1/GraspGPT_public/gcngrasp" 

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if GCNGRASP_DIR not in sys.path:
    sys.path.insert(0, GCNGRASP_DIR)

import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()

from visualize import draw_scene, get_gripper_control_points
from models.graspgpt_plain import GraspGPT_plain
from config import get_cfg_defaults
from data.SGNLoader import pc_normalize
from geometry_utils import regularize_pc_point_count

DEVICE = "cuda"

def encode_text(text, tokenizer, model, device, type=None):
    if type == 'od':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=300).to(device)
    elif type == 'td':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=200).to(device)
    elif type == 'li':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=21).to(device)
    else:
         raise ValueError(f'No such language embedding type: {type}')
    
    with torch.no_grad():
        output = model(**encoded_input)
        word_embedding = output[0]  
    
    return word_embedding[0], encoded_input['attention_mask'][0]

def test(model, pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask):   
    pc = pc.type(torch.cuda.FloatTensor)
    
    obj_desc = obj_desc.unsqueeze(0).to(DEVICE)
    obj_desc_mask = obj_desc_mask.unsqueeze(0).to(DEVICE)
    task_desc = task_desc.unsqueeze(0).to(DEVICE)
    task_desc_mask = task_desc_mask.unsqueeze(0).to(DEVICE)
    task_ins = task_ins.unsqueeze(0).to(DEVICE)
    task_ins_mask = task_ins_mask.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
    logits = logits.squeeze()
    
    if logits.dim() == 0: 
        logits = logits.unsqueeze(0)
        
    probs = torch.sigmoid(logits)
    return probs

def evaluate_grasps(in_data_path, out_data_path, task, obj_class):
    CKPT_PATH = "/home/zyp/pan1/GraspGPT_public/checkpoints/gcngrasp_split_mode_o_split_idx_0__2026-04-20-11-07/weights/best.ckpt"
    CFG_PATH = "/home/zyp/pan1/GraspGPT_public/cfg/eval/gcngrasp/gcngrasp_split_mode_o_split_idx_0_.yml"
    
    # 1. 加载 GraspGPT 模型
    cfg = get_cfg_defaults()
    cfg.merge_from_file(CFG_PATH)
    cfg.defrost()
    cfg.base_dir = os.path.join(BASE_DIR, 'data')
    cfg.freeze()

    model = GraspGPT_plain(cfg)
    print(f"[GPT-Worker] 加载权重: {CKPT_PATH}", flush=True)
    model_weights = torch.load(CKPT_PATH, map_location=DEVICE)
    state_dict = model_weights["state_dict"] if "state_dict" in model_weights else model_weights
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEVICE).eval()

    # 2. 加载 BERT 语言大模型 (彻底的本地物理路径模式)
    print("[GPT-Worker] 正在加载 BERT 语言模型...", flush=True)
    LOCAL_BERT = "/home/zyp/pan1/GraspGPT_public/local_bert"
    tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT)
    bert_model = BertModel.from_pretrained(LOCAL_BERT).to(DEVICE).eval()

    print(f"[GPT-Worker] 正在从官方数据库中匹配 '{obj_class}' 和 '{task}' 的原始描述...", flush=True)
    DB_BASE_DIR = "/home/zyp/pan1/GraspGPT_public/data/taskgrasp"
    
    mapped_obj = obj_class.replace('kitchen_knife', 'knife') 
    obj_db_path = os.path.join(DB_BASE_DIR, "obj_gpt_v2", mapped_obj, "descriptions", "0")
    
    if os.path.exists(obj_db_path):
        with open(os.path.join(obj_db_path, "all.txt"), 'r') as f:
            obj_desc_txt = f.read().strip()
        print(f" -> 匹配到物体描述: {obj_desc_txt[:50]}...")
    else:
        obj_desc_txt = f"This is a {obj_class}. It is used for various tasks."
        print(f" ⚠️ 警告：未找到 {mapped_obj} 的官方路径，使用退化模板。")

    task_db_path = os.path.join(DB_BASE_DIR, "task_gpt_v2", task, "descriptions", "0")
    
    if os.path.exists(task_db_path):
        with open(os.path.join(task_db_path, "all.txt"), 'r') as f:
            task_desc_txt = f.read().strip()
        print(f" -> 匹配到任务描述: {task_desc_txt[:50]}...")
    else:
        task_desc_txt = f"The task is to {task} objects."
        print(f" ⚠️ 警告：未找到 {task} 的任务路径，使用退化模板。")

    task_ins_txt = f"grasp the {obj_class.replace('_', ' ')} to {task}"

    obj_desc, obj_desc_mask = encode_text(obj_desc_txt, tokenizer, bert_model, DEVICE, type='od')
    task_desc, task_desc_mask = encode_text(task_desc_txt, tokenizer, bert_model, DEVICE, type='td')
    task_ins, task_ins_mask = encode_text(task_ins_txt, tokenizer, bert_model, DEVICE, type='li')
    
    cgn_data = np.load(in_data_path, allow_pickle=True)
    grasps_orig = cgn_data['grasps']
    pc_scene = cgn_data['pc']

    if len(grasps_orig) == 0:
        np.savez(out_data_path, success=False)
        return

    pc_mean = pc_scene[:, :3].mean(axis=0)
    pc_centered = pc_scene.copy()
    pc_centered[:, :3] -= pc_mean
    
    grasps_centered = grasps_orig.copy()
    grasps_centered[:, :3, 3] -= pc_mean

    pc_input = regularize_pc_point_count(pc_centered, cfg.num_points, use_farthest_point=False)

    print(f"[GPT-Worker] 正在对 {len(grasps_centered)} 个抓取进行打分...")
    probs = []
    
    for i in range(len(grasps_centered)):
        grasp = grasps_centered[i]
        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T[:, :3]
        
        pc = pc_input[:, :3]
        latent = np.concatenate([np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        latent = np.expand_dims(latent, axis=1)
        pc_combined = np.concatenate([pc, grasp_pc], axis=0) 

        pc_norm, grasp_norm = pc_normalize(pc_combined, grasp, pc_scaling=cfg.pc_scaling)
        pc_final = np.concatenate([pc_norm, latent], axis=1) 
        pc_tensor = torch.tensor([pc_final])

        prob = test(model, pc_tensor, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
        probs.append(prob.item())

    probs = np.array(probs)
    K = min(15, len(grasps_orig))
    topk_inds = probs.argsort()[-K:][::-1]
    
    best_probs = probs[topk_inds]
    best_grasps_centered = grasps_centered[topk_inds]
    best_grasps_orig = grasps_orig[topk_inds]

    print(f"\n[GPT-Worker] 🏆 最佳匹配概率: {best_probs[0]:.4f}")

    # === 在完全自动化跑测时，必须注释掉 Open3D 绘图逻辑 ===
    # grasp_colors = np.stack([np.ones(K) - best_probs, best_probs, np.zeros(K)], axis=1)
    # R_align = np.array([[ 0,  0, -1,  0], [ 0,  1,  0,  0], [ 1,  0,  0,  0], [ 0,  0,  0,  1]], dtype=np.float32)
    # R_roll = np.array([[ 1,  0,  0,  0], [ 0,  0, -1,  0], [ 0,  1,  0,  0], [ 0,  0,  0,  1]], dtype=np.float32)
    # visual_fix_matrix = np.matmul(R_align, R_roll)
    # best_grasps_vis = np.matmul(best_grasps_centered, visual_fix_matrix)
    # draw_scene( ... )

    np.savez(out_data_path, best_grasp=best_grasps_orig[0], score=best_probs[0], success=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_data', type=str, required=True)
    parser.add_argument('--out_data', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--obj_class', type=str, required=True)
    args = parser.parse_args()
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    evaluate_grasps(args.in_data, args.out_data, args.task, args.obj_class)
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from util import loss_func, utils_utils as utils
from util.opt import Options
from util.grab import Grab
from model_others.EAI import GCN_EAI

from helper import get_dct, get_idct, get_dct_norm, get_idct_norm

logger = get_logger(__name__)

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Parse options
    opt = Options().parse()
    
    # Set seed for reproducibility
    set_seed(opt.seed)
    
    # Initialize parameters
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr

    # Load datasets
    logger.info("Loading train data")
    train_dataset = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=0, 
                         debug=opt.is_debug, using_saved_file=opt.is_using_saved_file, using_noTpose2=opt.is_using_noTpose2)
    logger.info("Loading validation data")
    val_dataset = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=1, 
                       debug=opt.is_debug, using_saved_file=opt.is_using_saved_file, using_noTpose2=opt.is_using_noTpose2)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.train_batch, shuffle=False)

    logger.info(f"Train data: {len(train_dataset)}")
    logger.info(f"Validation data: {len(val_dataset)}")

    # Create model
    logger.info("Creating model")
    model = GCN_EAI(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout, 
                    num_stage=opt.num_stage, lh_node_n=opt.num_lh*3, rh_node_n=opt.num_rh*3, b_node_n=opt.num_body*3)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Load checkpoint if specified
    if opt.is_load:
        load_checkpoint(opt, model, optimizer, accelerator)

    # Training loop
    for epoch in range(start_epoch, opt.epochs):
        train_loss = train(train_loader, model, optimizer, accelerator, opt)
        val_loss = validate(val_loader, model, accelerator, opt)

        accelerator.print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if accelerator.is_main_process:
            is_best = val_loss < err_best
            err_best = min(val_loss, err_best)
            save_checkpoint(opt, epoch, model, optimizer, train_loss, is_best, accelerator)

        # Learning rate decay
        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

def train(train_loader, model, optimizer, accelerator, opt):
    model.train()
    total_loss = 0
    
    for input_pose, target_pose in tqdm(train_loader, disable=not accelerator.is_local_main_process):
        optimizer.zero_grad()
        
        model_input = dct_trans(input_pose, opt.is_hand_norm)
        out_pose, mmdloss_ab, mmdloss_ac, mmdloss_bc = model(model_input)
        pred_3d, targ_3d = idct_trans(out_pose, target_pose, opt.is_hand_norm, accelerator.device)
        
        loss = compute_loss(pred_3d, targ_3d, opt, mmdloss_ab, mmdloss_ac, mmdloss_bc)
        
        accelerator.backward(loss)
        if opt.max_norm:
            accelerator.clip_grad_norm_(model.parameters(), opt.max_norm)
        optimizer.step()
        
        total_loss += loss.item() * input_pose.size(0)
    
    return total_loss / len(train_loader.dataset)

def validate(val_loader, model, accelerator, opt):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_pose, target_pose in val_loader:
            model_input = dct_trans(input_pose, opt.is_hand_norm)
            out_pose, _, _, _ = model(model_input)
            pred_3d, targ_3d = idct_trans(out_pose, target_pose, opt.is_hand_norm, accelerator.device)
            
            loss = loss_func.joint_loss(pred_3d, targ_3d)
            total_loss += loss.item() * input_pose.size(0)
    
    return total_loss / len(val_loader.dataset)

def compute_loss(pred_3d, targ_3d, opt, mmdloss_ab, mmdloss_ac, mmdloss_bc):
    if opt.is_weighted_jointloss:
        loss_jt = loss_func.weighted_joint_loss(pred_3d, targ_3d, ratio=0.6)
    else:
        loss_jt = loss_func.joint_loss(pred_3d, targ_3d)
        loss_pjt = loss_func.relative_hand_loss(pred_3d, targ_3d)
    
    if opt.is_boneloss:
        loss_bl = loss_func.bone_loss(pred_3d, targ_3d)
        loss = loss_jt + 0.1 * loss_bl + 0.1 * loss_pjt
    else:
        loss = loss_jt
    
    loss = loss + 0.001 * (mmdloss_ab + mmdloss_ac + mmdloss_bc)
    return loss

def dct_trans(input_pose, is_hand_norm):
    if is_hand_norm:
        return get_dct_norm(input_pose)
    else:
        return get_dct(input_pose)

def idct_trans(out_pose, target_pose, is_hand_norm, device):
    if is_hand_norm:
        return get_idct_norm(out_pose, target_pose, device)
    else:
        return get_idct(out_pose, target_pose, device)

def load_checkpoint(opt, model, optimizer, accelerator):
    ckpt_path = f"{opt.ckpt}/ckpt_{opt.model_type}_best.pth.tar"
    logger.info(f"Loading checkpoint from {ckpt_path}")
    
    ckpt = accelerator.load_state_dict(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    
    return ckpt['epoch'], ckpt['err_best'], ckpt['lr']

def save_checkpoint(opt, epoch, model, optimizer, loss, is_best, accelerator):
    ckpt_dir = f"{opt.ckpt}/{opt.model_type}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    state = {
        'epoch': epoch + 1,
        'lr': optimizer.param_groups[0]['lr'],
        'err_best': loss,
        'state_dict': accelerator.get_state_dict(model),
        'optimizer': optimizer.state_dict()
    }
    
    ckpt_path = f"{ckpt_dir}/ckpt_epoch_{epoch+1}.pth.tar"
    
    # Save the state dictionary
    accelerator.save(state, ckpt_path)
    
    if is_best:
        best_path = f"{ckpt_dir}/ckpt_best.pth.tar"
        # Copy the best model
        if accelerator.is_main_process:
            shutil.copyfile(ckpt_path, best_path)

    # Save the Accelerator state
    accelerator.save_state(ckpt_dir)

if __name__ == "__main__":
    main()
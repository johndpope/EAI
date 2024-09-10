from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import numpy as np
import torch
from torch.utils.data import DataLoader
from util.grab import Grab
from model_others.EAI import GCN_EAI
from util.opt import Options
import os
from util import loss_func
from helper import get_dct_norm,get_idct_norm,get_dct,get_idct
logger = get_logger(__name__)

def main(opt):
    # Initialize accelerator
    accelerator = Accelerator()

    # Set seed for reproducibility
    set_seed(opt.seed)

    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n

    logger.info("Creating model")
    model = GCN_EAI(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout, 
                    num_stage=opt.num_stage, lh_node_n=opt.num_lh*3, rh_node_n=opt.num_rh*3, b_node_n=opt.num_body*3)

    dct_trans = get_dct_norm if opt.is_hand_norm else get_dct
    idct_trans = get_idct_norm if opt.is_hand_norm else get_idct

    # Load checkpoint
    train_ckpt_path = './checkpoint/LTD/ckpt_best.pth.tar'

    if not os.path.isfile(train_ckpt_path):
        logger.error(f"Checkpoint file not found: {train_ckpt_path}")
        logger.info("Please check the following:")
        logger.info("1. Ensure the path to the checkpoint file is correct.")
        logger.info("2. Verify that the checkpoint file exists in the specified location.")
        logger.info("3. Check if you have the necessary permissions to access the file.")
        raise FileNotFoundError(f"Checkpoint file not found: {train_ckpt_path}")

    logger.info(f"Loading checkpoint from '{train_ckpt_path}'")
    try:
        # Use torch.load instead of accelerator.load_state
        ckpt = torch.load(train_ckpt_path, map_location=accelerator.device)
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        raise
    start_epoch = ckpt['epoch']
    err_best = ckpt['err_best']
    lr = ckpt['lr']

    # Load model state dict
    model.load_state_dict(ckpt['state_dict'])



    logger.info(f"Checkpoint loaded (epoch: {start_epoch} | err: {err_best} | lr: {lr})")


    # Prepare test dataset
    test_dataset = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=2, 
                        debug=opt.is_debug, using_saved_file=opt.is_using_saved_file, using_noTpose2=opt.is_using_noTpose2)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch, shuffle=False, num_workers=0, pin_memory=True)

    # Prepare for distributed evaluation
    model, test_loader = accelerator.prepare(model, test_loader)

    # Evaluate
    test_results = test_split(test_loader, model, accelerator.device, dct_trans, idct_trans)
    test_loss, test_body_loss, test_lhand_loss, test_rhand_loss, test_lhand_rel_loss, test_rhand_rel_loss, _ = test_results

    eval_frame = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    logger.info('>>> Frames  |>>> whole body |>>> main body |>>> left hand |>>> right hand|>>> rel left hand |>>> rel right hand')
    for i, f in enumerate(eval_frame):
        logger.info(f'>>> {f}       |>>> {test_loss[i]:.3f}     |>>> {test_body_loss[i]:.3f}     |>>> {test_lhand_loss[i]:.3f}     |>>> {test_rhand_loss[i]:.3f}     |>>> {test_lhand_rel_loss[i]:.3f}         |>>> {test_rhand_rel_loss[i]:.3f}')

def test_split(test_loader, model, device, dct_trans, idct_trans):
    model.eval()
    N = 0
    eval_frame = [32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
    t_posi = np.zeros(len(eval_frame))
    t_body_posi = np.zeros(len(eval_frame))
    t_lhand_posi = np.zeros(len(eval_frame))
    t_rhand_posi = np.zeros(len(eval_frame))
    t_lhand_rel_posi = np.zeros(len(eval_frame))
    t_rhand_rel_posi = np.zeros(len(eval_frame))

    with torch.no_grad():
        for input_pose, target_pose in test_loader:
            model_input = dct_trans(input_pose)
            n = input_pose.shape[0]
            
            out_pose, _, _, _ = model(model_input)
            pred_3d, targ_3d = idct_trans(y_out=out_pose, out_joints=target_pose, device=device)

            rel_pred_3d = pred_3d.clone()
            rel_targ_3d = targ_3d.clone()

            rel_pred_3d[:,:,25:40] -= rel_pred_3d[:,:,20:21]
            rel_pred_3d[:,:,40:] -= rel_pred_3d[:,:,21:22]
            rel_targ_3d[:,:,25:40] -= rel_targ_3d[:,:,20:21]
            rel_targ_3d[:,:,40:] -= rel_targ_3d[:,:,21:22]

            for k, j in enumerate(eval_frame):
                test_out, test_joints = pred_3d[:, j, :, :], targ_3d[:, j, :, :]
                loss_wholebody, _ = loss_func.joint_body_loss_test(test_out, test_joints)
                t_posi[k] += loss_wholebody.cpu().data.numpy() * n * 100

                test_body_out, test_body_joints = pred_3d[:, j, :25, :], targ_3d[:, j, :25, :]
                t_body_posi[k] += loss_func.joint_loss(test_body_out, test_body_joints).cpu().data.numpy() * n * 100

                test_lhand_out, test_lhand_joints = pred_3d[:, j, 25:40, :], targ_3d[:, j, 25:40, :]
                t_lhand_posi[k] += loss_func.joint_loss(test_lhand_out, test_lhand_joints).cpu().data.numpy() * n * 100

                test_rhand_out, test_rhand_joints = pred_3d[:, j, 40:, :], targ_3d[:, j, 40:, :]
                t_rhand_posi[k] += loss_func.joint_loss(test_rhand_out, test_rhand_joints).cpu().data.numpy() * n * 100

                test_lhand_rel_out, test_lhand_rel_joints = rel_pred_3d[:, j, 25:40, :], rel_targ_3d[:, j, 25:40, :]
                t_lhand_rel_posi[k] += loss_func.joint_loss(test_lhand_rel_out, test_lhand_rel_joints).cpu().data.numpy() * n * 100

                test_rhand_rel_out, test_rhand_rel_joints = rel_pred_3d[:, j, 40:, :], rel_targ_3d[:, j, 40:, :]
                t_rhand_rel_posi[k] += loss_func.joint_loss(test_rhand_rel_out, test_rhand_rel_joints).cpu().data.numpy() * n * 100
            
            N += n

    return t_posi / N, t_body_posi / N, t_lhand_posi / N, t_rhand_posi / N, t_lhand_rel_posi / N, t_rhand_rel_posi / N, N

if __name__ == "__main__":
    option = Options().parse()
    main(option)
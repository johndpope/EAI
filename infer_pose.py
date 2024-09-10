from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.grab import Grab
from model_others.EAI import GCN_EAI
from util.opt import Options
import os
from util import loss_func
from helper import get_dct_norm,get_idct_norm,get_dct,get_idct,preprocess_for_eai,postprocess_eai_output
logger = get_logger(__name__)



def run_eai_model(model, input_sequence, device):
    # Ensure the input is on the correct device
    input_sequence = input_sequence.to(device)
    
    model_input = get_dct_norm(input_sequence)
    model_input = model_input.to(device)
    
    # Ensure the model is on the correct device
    model = model.to(device)
    
    with torch.no_grad():
        output, _, _, _ = model(model_input)
    return output


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = model.state_dict()
    
    # Filter out unnecessary keys
    pretrained_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
    
    # Resize tensors if necessary
    for k, v in pretrained_state_dict.items():
        if v.size() != model_state_dict[k].size():
            print(f"Mismatched size for {k}: checkpoint {v.size()} vs model {model_state_dict[k].size()}")
            
            if 'att' in k and v.dim() == 2:
                # Handle attention matrix resizing
                new_v = torch.zeros(model_state_dict[k].size(), device=v.device)
                min_rows = min(v.size(0), new_v.size(0))
                min_cols = min(v.size(1), new_v.size(1))
                new_v[:min_rows, :min_cols] = v[:min_rows, :min_cols]
                
                # If new matrix is larger, we need to ensure it's still a valid attention matrix
                if new_v.size(0) > v.size(0) or new_v.size(1) > v.size(1):
                    new_v = new_v / new_v.sum(dim=1, keepdim=True)
                
                pretrained_state_dict[k] = new_v
            elif v.dim() == 1:
                # For 1D tensors (like biases), we can use simple resizing
                new_v = torch.zeros(model_state_dict[k].size(), device=v.device)
                new_v[:min(v.size(0), new_v.size(0))] = v[:min(v.size(0), new_v.size(0))]
                pretrained_state_dict[k] = new_v
            else:
                print(f"Cannot resize tensor {k} from {v.size()} to {model_state_dict[k].size()}")
                # Initialize randomly if we can't resize
                pretrained_state_dict[k] = torch.randn(model_state_dict[k].size(), device=v.device)
    
    # Update model state dict
    model_state_dict.update(pretrained_state_dict)
    
    # Load the updated state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    print(f"Checkpoint loaded. Some layers have been resized to fit the current model architecture.")
    return checkpoint['epoch'], checkpoint.get('err_best', checkpoint.get('train_loss')), checkpoint.get('lr', None)


def main(opt):
    # Initialize accelerator
    accelerator = Accelerator()

    # Set seed for reproducibility
    set_seed(opt.seed)

    # Load JSON data for two poses
    with open('body1.json', 'r') as f:
        json_data1 = json.load(f)
    with open('body2.json', 'r') as f:
        json_data2 = json.load(f)

    # Load EAI model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n


 # Load checkpoint
    train_ckpt_path = './checkpoint/LTD/ckpt_best.pth.tar'



    logger.info("Creating model")
   # model = GCN_EAI(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout, 
    #                num_stage=opt.num_stage, lh_node_n=opt.num_lh*3, rh_node_n=opt.num_rh*3, b_node_n=opt.num_body*3)

    model = GCN_EAI(input_feature=60, hidden_feature=256, p_dropout=0.5, num_stage=12, 
                    lh_node_n=48, rh_node_n=48, b_node_n=75)
    
    start_epoch, err_best, lr_now = load_checkpoint(model, train_ckpt_path, accelerator.device)
    print(f"Checkpoint loaded (epoch: {start_epoch} | err: {err_best} | lr: {lr_now})")

    # Preprocess JSON data
    input_sequence = preprocess_for_eai(json_data1, json_data2)
    
    # Run EAI model
    output = run_eai_model(model, input_sequence, device)
    
    # Postprocess output
    animated_sequence = postprocess_eai_output(output, device)
    
    # Here you would save or visualize the animated_sequence
    print(f"Generated animated sequence with shape: {animated_sequence.shape}")
    # The shape should be (output_n, 55, 3), where output_n is typically 30
    # This sequence represents the predicted motion from the first pose to the second and beyond

    # Initialize accelerator
    accelerator = Accelerator()

    # Set seed for reproducibility
    set_seed(opt.seed)


    dct_trans = get_dct_norm if opt.is_hand_norm else get_dct
    idct_trans = get_idct_norm if opt.is_hand_norm else get_idct

   
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
    print('test_results:',test_results)
    test_loss, test_body_loss, test_lhand_loss, test_rhand_loss, test_lhand_rel_loss, test_rhand_rel_loss, _ = test_results

    eval_frame = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    print('>>> Frames  |>>> whole body |>>> main body |>>> left hand |>>> right hand|>>> rel left hand |>>> rel right hand')
    for i, f in enumerate(eval_frame):
        print(f'>>> {f}       |>>> {test_loss[i]:.3f}     |>>> {test_body_loss[i]:.3f}     |>>> {test_lhand_loss[i]:.3f}     |>>> {test_rhand_loss[i]:.3f}     |>>> {test_lhand_rel_loss[i]:.3f}         |>>> {test_rhand_rel_loss[i]:.3f}')

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
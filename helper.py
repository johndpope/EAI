import torch
import shutil
import random
import torch.optim
import numpy as np
import pandas as pd
import torch.nn as nn

# 相对plevis的坐标系下：3D转DCT
def get_dct(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    dct_m_in, _ = get_dct_matrix(frame)
    dct_m_in = dct_m_in.to(out_joints.device)  # Just move to the correct device
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = torch.matmul(dct_m_in[0:frame, :], input_joints)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

# 相对plevis的坐标系下：DCT转3D
def get_idct(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = idct_m.to(device)  # Just move to the correct device
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d

# 身体的关节，相对plevis的坐标系下：3D转DCT； 针对手部，相对wrist关节的坐标系下：3D转DCT；
def get_dct_norm(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    out_joints[:,:,25:40,:] = out_joints[:,:,25:40,:] - out_joints[:,:,20:21,:]
    out_joints[:,:,40:,:] = out_joints[:,:,40:,:] - out_joints[:,:,21:22,:]
    dct_m_in, _ = get_dct_matrix(frame)
    dct_m_in = dct_m_in.to(out_joints.device)  # Just move to the correct device
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = torch.matmul(dct_m_in[0:frame, :], input_joints)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

# 身体的关节，相对plevis的坐标系下：DCT转3D； 针对手部，相对wrist关节的坐标系下：DCT转3D；
def get_idct_norm(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = idct_m.to(device)  # Just move to the correct device
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    outputs_p3d[:,:,25:40,:] = outputs_p3d[:,:,25:40,:] + outputs_p3d[:,:,20:21,:]
    outputs_p3d[:,:,40:,:] = outputs_p3d[:,:,40:,:] + outputs_p3d[:,:,21:22,:]
    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d


# 一维DCT变换
def get_dct_matrix(N):
    dct_m = np.eye(N)  # 返回one-hot数组
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)  # 2/35开更
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return torch.from_numpy(dct_m).float(), torch.from_numpy(idct_m).float()


def map_coco_to_grab(coco_keypoints):
    # Define mapping from COCO Whole-Body to GRAB 75-joint structure
    mapping = {
        # Body (25 joints)
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
        
        # Left Hand (21 joints)
        'left_thumb4': 25, 'left_thumb3': 26, 'left_thumb2': 27, 'left_thumb_third_joint': 28,
        'left_forefinger4': 29, 'left_forefinger3': 30, 'left_forefinger2': 31, 'left_forefinger_third_joint': 32,
        'left_middle_finger4': 33, 'left_middle_finger3': 34, 'left_middle_finger2': 35, 'left_middle_finger_third_joint': 36,
        'left_ring_finger4': 37, 'left_ring_finger3': 38, 'left_ring_finger2': 39, 'left_ring_finger_third_joint': 40,
        'left_pinky_finger4': 41, 'left_pinky_finger3': 42, 'left_pinky_finger2': 43, 'left_pinky_finger_third_joint': 44,
        'left_wrist': 45,
        
        # Right Hand (21 joints)
        'right_thumb4': 50, 'right_thumb3': 51, 'right_thumb2': 52, 'right_thumb_third_joint': 53,
        'right_forefinger4': 54, 'right_forefinger3': 55, 'right_forefinger2': 56, 'right_forefinger_third_joint': 57,
        'right_middle_finger4': 58, 'right_middle_finger3': 59, 'right_middle_finger2': 60, 'right_middle_finger_third_joint': 61,
        'right_ring_finger4': 62, 'right_ring_finger3': 63, 'right_ring_finger2': 64, 'right_ring_finger_third_joint': 65,
        'right_pinky_finger4': 66, 'right_pinky_finger3': 67, 'right_pinky_finger2': 68, 'right_pinky_finger_third_joint': 69,
        'right_wrist': 70,
    }
    
    grab_keypoints = np.zeros((75, 3))
    for coco_name, grab_idx in mapping.items():
        if coco_name in coco_keypoints:
            grab_keypoints[grab_idx] = coco_keypoints[coco_name][:3]  # Use x, y, confidence
    
    # Fill in missing joints with interpolated or estimated values
    for i in range(75):
        if np.all(grab_keypoints[i] == 0):
            nearest_non_zero = np.nonzero(np.any(grab_keypoints != 0, axis=1))[0]
            if len(nearest_non_zero) > 0:
                nearest_idx = nearest_non_zero[np.argmin(np.abs(nearest_non_zero - i))]
                grab_keypoints[i] = grab_keypoints[nearest_idx]
    
    return grab_keypoints

# Update preprocess_for_eai function to remove the separate depth estimation step
def preprocess_for_eai(json_data1, json_data2, input_n=30):
    # Process both poses
    coco_keypoints1 = json_data1[0]
    coco_keypoints2 = json_data2[0]
    
    grab_keypoints_3d1 = map_coco_to_grab(coco_keypoints1)
    grab_keypoints_3d2 = map_coco_to_grab(coco_keypoints2)
    
    # Interpolate between the two poses
    sequence = interpolate_poses(grab_keypoints_3d1, grab_keypoints_3d2, input_n)
    
    # Reshape the sequence to match the expected input size
    sequence = sequence.reshape(1, input_n, 75, 3)
    
    print("Sequence shape after interpolation:", sequence.shape)
    print("Sequence shape after reshaping:", sequence.shape)
    
    tensor_sequence = torch.tensor(sequence).float()
    print("Tensor sequence shape:", tensor_sequence.shape)
    
    return tensor_sequence  # This will be on CPU

def estimate_depth(keypoints_2d):
    # Simple depth estimation (this is a placeholder and would need a more sophisticated approach)
    depth = np.ones(keypoints_2d.shape[0]) * 10  # Assume all points are 10 units away
    return np.column_stack((keypoints_2d, depth))

def generate_sequence(initial_pose, num_frames=30):
    # Generate a sequence by interpolating to a neutral pose
    neutral_pose = np.zeros_like(initial_pose)
    sequence = np.array([initial_pose * (1 - i/num_frames) + neutral_pose * (i/num_frames) 
                         for i in range(num_frames)])
    return sequence

def interpolate_poses(pose1, pose2, num_frames):
    return np.array([pose1 * (1 - i/(num_frames-1)) + pose2 * (i/(num_frames-1)) 
                     for i in range(num_frames)])



def postprocess_eai_output(output, device):
    pred_3d, _ = get_idct_norm(output, output, device)
    return pred_3d.squeeze(0).cpu().numpy()



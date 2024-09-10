import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================

        self.parser.add_argument('--device', type=str, default='cuda:0',
                                 help='path to amass Synthetic dataset')
        self.parser.add_argument('--grab_data_dict', type=str, default='./Dataset_GRAB/',help='path to GRAB dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')
        self.parser.add_argument('--model_type', type=str, default='LTD', help='path to save checkpoint')


        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true',
                                 help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=12, help='# layers in linear model')
        self.parser.add_argument('--num_body', type=int, default=25, help='# layers in linear model')
        self.parser.add_argument('--num_lh', type=int, default=15, help='# layers in linear model')
        self.parser.add_argument('--num_rh', type=int, default=15, help='# layers in linear model')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=1.0e-3)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=30, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=30, help='future seq length')
        self.parser.add_argument('--all_n', type=int, default=60, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=64)
        self.parser.add_argument('--val_batch', type=int, default=128)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=0, help='subprocesses to use for data loading')
        self.parser.add_argument('--seed', type=int, default=1024, help='random seed')
        self.parser.add_argument("--local_rank", type=int, help="local rank")
        self.parser.add_argument('--W_pg', type=float, default=0.6, help='The weight of information propagation between part') 
        self.parser.add_argument('--W_p', type=float, default=0.6, help='The weight of part on the whole body')

        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        self.parser.add_argument('--is_debug', dest='is_debug', action='store_true', help='wether to debug')
        self.parser.add_argument('--is_exp', dest='is_exp', action='store_true', help='wether to save different model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true',
                                 help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true',
                                 help='whether to normalize the angles/3d coordinates')
        self.parser.add_argument('--is_using_saved_file', dest='is_using_saved_file', action='store_true',
                                 help='whether to normalize the angles/3d coordinates')
        self.parser.add_argument('--is_hand_norm', dest='is_hand_norm', action='store_true',help='')
        self.parser.add_argument('--is_hand_norm_split', dest='is_hand_norm_split', action='store_true',help='')
        self.parser.add_argument('--is_part', dest='is_part', action='store_true', help='')
        self.parser.add_argument('--part_type', type=str, default='lhand', help='')
        self.parser.add_argument('--is_boneloss', dest='is_boneloss', action='store_true', help='')
        self.parser.add_argument('--is_weighted_jointloss', dest='is_weighted_jointloss', action='store_true', help='')
        self.parser.add_argument('--is_using_noTpose2', dest='is_using_noTpose2', action='store_true', help='')
        self.parser.add_argument('--is_using_raw', dest='is_using_raw', action='store_true', help='')

        self.parser.add_argument('--J', type=int, default=1, help='The number of wavelet filters')                    
        self.parser.set_defaults(max_norm=True)


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()

        return self.opt

import torch
import os


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma): 
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_ckpt(state, ckpt_path, is_best=True, file_name=['ckpt_best.pth.tar', 'ckpt_last.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[1])
    torch.save(state, file_path)
    if is_best:
        file_path = os.path.join(ckpt_path, file_name[0])
        torch.save(state, file_path)

import numpy as np
import torch
import logging
from copy import copy
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def prepare_params(params, frame_mask, dtype=np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def DotDict(in_dict):
    out_dict = copy(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def append2dict(source, data):
    for k in data.keys():
        if isinstance(data[k], list):
            source[k] += data[k].astype(np.float32)
        else:
            source[k].append(data[k].astype(np.float32))


def np2torch(item):
    out = {}
    for k, v in item.items():
        if v == []:
            continue
        if isinstance(v, list):
            try:
                out[k] = torch.from_numpy(np.concatenate(v))
            except:
                out[k] = torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v)
    return out


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array


def makepath(desired_path, isfile=False):
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def makelogger(log_dir, mode='w'):
    makepath(log_dir, isfile=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler('%s' % log_dir, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def euler(rots, order='xyz', units='deg'):
    rots = np.asarray(rots)
    single_val = False if len(rots.shape) > 1 else True
    rots = rots.reshape(-1, 3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz, order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis == 'x':
                r = np.dot(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), r)
            if axis == 'y':
                r = np.dot(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), r)
            if axis == 'z':
                r = np.dot(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats


def create_video(path, fps=30, name='movie'):
    import os
    import subprocess

    src = os.path.join(path, '%*.png')
    movie_path = os.path.join(path, '%s.mp4' % name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path, '%s_%02d.mp4' % (name, i))
        i += 1

    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)

    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue

import torch
import os


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma): 
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_ckpt(state, ckpt_path, file_name=['ckpt_best.pth.tar']):
    file_path = os.path.join(ckpt_path, file_name[0])
    torch.save(state, file_path)


contact_ids = {'Body': 1,
               'L_Thigh': 2,
               'R_Thigh': 3,
               'Spine': 4,
               'L_Calf': 5,
               'R_Calf': 6,
               'Spine1': 7,
               'L_Foot': 8,
               'R_Foot': 9,
               'Spine2': 10,
               'L_Toes': 11,
               'R_Toes': 12,
               'Neck': 13,
               'L_Shoulder': 14,
               'R_Shoulder': 15,
               'Head': 16,
               'L_UpperArm': 17,
               'R_UpperArm': 18,
               'L_ForeArm': 19,
               'R_ForeArm': 20,
               'L_Hand': 21,
               'R_Hand': 22,
               'Jaw': 23,
               'L_Eye': 24,
               'R_Eye': 25,
               'L_Index1': 26,
               'L_Index2': 27,
               'L_Index3': 28,
               'L_Middle1': 29,
               'L_Middle2': 30,
               'L_Middle3': 31,
               'L_Pinky1': 32,
               'L_Pinky2': 33,
               'L_Pinky3': 34,
               'L_Ring1': 35,
               'L_Ring2': 36,
               'L_Ring3': 37,
               'L_Thumb1': 38,
               'L_Thumb2': 39,
               'L_Thumb3': 40,
               'R_Index1': 41,
               'R_Index2': 42,
               'R_Index3': 43,
               'R_Middle1': 44,
               'R_Middle2': 45,
               'R_Middle3': 46,
               'R_Pinky1': 47,
               'R_Pinky2': 48,
               'R_Pinky3': 49,
               'R_Ring1': 50,
               'R_Ring2': 51,
               'R_Ring3': 52,
               'R_Thumb1': 53,
               'R_Thumb2': 54,
               'R_Thumb3': 55}


def normal_init_(layer, mean_, sd_, bias, norm_bias=True):
    """Intialization of layers with normal distribution with mean and bias"""
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        layer.weight.data.normal_(mean_, sd_)
        if norm_bias:
            layer.bias.data.normal_(bias, 0.05)
        else:
            layer.bias.data.fill_(bias)


def weight_init(
        module,
        mean_=0,
        sd_=0.004,
        bias=0.0,
        norm_bias=False,
        init_fn_=normal_init_):
    """Initialization of layers with normal distribution"""
    moduleclass = module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    init_fn_(l, mean_, sd_, bias, norm_bias)
            else:
                init_fn_(layer, mean_, sd_, bias, norm_bias)
    except TypeError:
        init_fn_(module, mean_, sd_, bias, norm_bias)


def xavier_init_(layer, mean_, sd_, bias, norm_bias=True):
    classname = layer.__class__.__name__
    if classname.find('Linear') != -1:
        print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
        nn.init.xavier_uniform_(layer.weight.data)
        if norm_bias:
            layer.bias.data.normal_(0, 0.05)
        else:
            layer.bias.data.zero_()


def create_dir_tree(base_dir):
    dir_tree = ['models', 'tf_logs', 'config', 'std_log']
    for dir_ in dir_tree:
        os.makedirs(os.path.join(base_dir, dir_), exist_ok=True)


def create_look_ahead_mask(seq_length, is_nonautoregressive=False):
    """Generates a binary mask to prevent to use future context in a sequence."""
    if is_nonautoregressive:
        return np.zeros((seq_length, seq_length), dtype=np.float32)
    x = np.ones((seq_length, seq_length), dtype=np.float32)
    mask = np.triu(x, 1).astype(np.float32)
    return mask  # (seq_len, seq_len)


RED = (0, 1, 1)
ORANGE = (20/360, 1, 1)
YELLOW = (60/360, 1, 1)
GREEN = (100/360, 1, 1)
CYAN = (175/360, 1, 1)
BLUE = (210/360, 1, 1)

RED_DARKER = (0, 1, 0.25)
ORANGE_DARKER = (20/360, 1, 0.25)
YELLOW_DARKER = (60/360, 1, 0.25)
GREEN_DARKER = (100/360, 1, 0.25)
CYAN_DARKER = (175/360, 1, 0.25)
BLUE_DARKER = (210/360, 1, 0.25)
class Grab_Skeleton_55:
    num_joints = 55
    start_joints = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 18, 17, 19, 15, 15, 15, 
                    21, 21, 21, 21, 21, 52, 53, 40, 41, 43, 44, 49, 50, 46, 47, 
                    20, 20, 20, 20, 20, 37, 38, 25, 26, 28, 29, 34, 35, 31, 32] 
    end_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 19, 21, 22, 23, 24,  
                  40, 43, 49, 46, 52, 53, 54, 41, 42, 44, 45, 50, 51, 47, 48,  
                  37, 25, 28, 34, 31, 38, 39, 26, 27, 29, 30, 35, 36, 32, 33,  
                  ]
    bones = list(zip(start_joints,end_joints))


def define_actions(action='all'):
    if action == 'all':
        return ['airplane-fly-1', 'airplane-lift-1', 'airplane-pass-1', 'alarmclock-lift-1', 'alarmclock-pass-1',
                'alarmclock-see-1', 'apple-eat-1', 'apple-pass-1', 'banana-eat-1', 'banana-lift-1', 'banana-pass-1',
                'banana-peel-1', 'banana-peel-2', 'binoculars-lift-1', 'binoculars-pass-1', 'binoculars-see-1',
                'bowl-drink-1',
                'bowl-drink-2', 'bowl-lift-1', 'bowl-pass-1', 'camera-browse-1', 'camera-pass-1',
                'camera-takepicture-1',
                'camera-takepicture-2', 'camera-takepicture-3', 'cubelarge-inspect-1', 'cubelarge-lift-1',
                'cubelarge-pass-1',
                'cubemedium-inspect-1', 'cubemedium-lift-1', 'cubemedium-pass-1', 'cubesmall-inspect-1',
                'cubesmall-lift-1',
                'cubesmall-pass-1', 'cup-drink-1', 'cup-drink-2', 'cup-lift-1', 'cup-pass-1', 'cup-pour-1',
                'cylinderlarge-inspect-1', 'cylinderlarge-lift-1', 'cylinderlarge-pass-1', 'cylindermedium-inspect-1',
                'cylindermedium-pass-1', 'cylindersmall-inspect-1', 'cylindersmall-pass-1', 'doorknob-lift-1',
                'doorknob-use-1', 'doorknob-use-2', 'duck-pass-1', 'elephant-inspect-1', 'elephant-pass-1',
                'eyeglasses-wear-1', 'flashlight-on-1', 'flashlight-on-2', 'flute-pass-1', 'flute-play-1',
                'fryingpan-cook-1',
                'fryingpan-cook-2', 'gamecontroller-lift-1', 'gamecontroller-pass-1', 'gamecontroller-play-1',
                'hammer-lift-1',
                'hammer-pass-1', 'hammer-use-1', 'hammer-use-2', 'hammer-use-3', 'hand-inspect-1', 'hand-lift-1',
                'hand-pass-1', 'hand-shake-1', 'headphones-lift-1', 'headphones-pass-1', 'headphones-use-1',
                'knife-chop-1',
                'knife-pass-1', 'knife-peel-1', 'lightbulb-pass-1', 'lightbulb-screw-1', 'mouse-lift-1', 'mouse-pass-1',
                'mouse-use-1', 'mug-drink-1', 'mug-drink-2', 'mug-lift-1', 'mug-pass-1', 'mug-toast-1', 'phone-call-1',
                'phone-lift-1', 'phone-pass-1', 'piggybank-pass-1', 'piggybank-use-1', 'pyramidlarge-pass-1',
                'pyramidmedium-inspect-1', 'pyramidmedium-lift-1', 'pyramidmedium-pass-1', 'pyramidsmall-inspect-1',
                'scissors-pass-1', 'scissors-use-1', 'spherelarge-inspect-1', 'spherelarge-lift-1',
                'spherelarge-pass-1',
                'spheremedium-inspect-1', 'spheremedium-lift-1', 'spheremedium-pass-1', 'spheresmall-inspect-1',
                'spheresmall-pass-1', 'stamp-lift-1', 'stamp-pass-1', 'stamp-stamp-1', 'stanfordbunny-inspect-1',
                'stanfordbunny-lift-1', 'stanfordbunny-pass-1', 'stapler-lift-1', 'stapler-pass-1', 'stapler-staple-1',
                'stapler-staple-2', 'teapot-pass-1', 'teapot-pour-1', 'teapot-pour-2', 'toothpaste-lift-1',
                'toothpaste-pass-1', 'toothpaste-squeeze-1', 'toothpaste-squeeze-2', 'toruslarge-inspect-1',
                'toruslarge-lift-1', 'toruslarge-pass-1', 'torusmedium-inspect-1', 'torusmedium-lift-1',
                'torusmedium-pass-1',
                'torussmall-inspect-1', 'torussmall-lift-1', 'torussmall-pass-1', 'train-lift-1', 'train-pass-1',
                'train-play-1', 'watch-pass-1', 'waterbottle-drink-1', 'waterbottle-pass-1', 'waterbottle-pour-1',
                'wineglass-drink-1', 'wineglass-drink-2', 'wineglass-lift-1', 'wineglass-pass-1', 'wineglass-toast-1']
    else:
        return action

if __name__ == "__main__":
    skeleton = Grab_Skeleton_55
    print(skeleton.bones)

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class Grab(Dataset):
    def __init__(self, path_to_data, input_n, output_n, split=0, using_saved_file=True, using_noTpose2=False, norm=True, debug=False,opt=None,using_raw=False):
        tra_val_test = ['train', 'val', 'test']
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)

        data_size = {}
        data_size[0] = (176384, 60, 55, 3)
        data_size[1] = (52255, 60, 55, 3)


        if split==2:
            sampled_seq = np.load('{}/grab_{}.npy'.format(path_to_data,tra_val_test[split]))
        else:
            # print('>>> remove the first and last 1 second')
            # sampled_seq = np.load('{}/grab_dataloader_normalized_noTpose2_{}.npy'.format(path_to_data,tra_val_test[split]))
            tmp_bin_size = data_size[split]
            tmp_seq = np.memmap('{}/grab_dataloader_normalized_noTpose2_{}.bin'.format(path_to_data,tra_val_test[split]), dtype=np.float32, shape=tmp_bin_size)
            tem_res = np.frombuffer(tmp_seq, dtype=np.float32)
            sampled_seq= tem_res.reshape(tmp_bin_size)

        self.input_pose = torch.from_numpy(sampled_seq[:, i_idx])
        print("input", self.input_pose.shape)
        self.target_pose = torch.from_numpy(sampled_seq)
        print("target", self.target_pose.shape)

        import gc
        del sampled_seq
        gc.collect()
        return

    def gen_data(self):
        for input in self.input_pose:
            batch_samples = []
            while len(batch_samples) > 0:
                yield batch_samples.pop()
    def __len__(self):
        return np.shape(self.input_pose)[0]

    def __getitem__(self, item):
        return self.input_pose[item], self.target_pose[item]



import torch
import numpy as np
from .data_utils import Grab_Skeleton_55


def poses_loss(y_out, out_poses):
    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    loss = torch.mean(torch.norm(y_out - out_poses, 2, 1))
    return loss

def joint_loss(y_out, out_poses):
    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    return torch.mean(torch.norm(y_out - out_poses, 2, 1))

def relative_hand_loss(y_out, out_poses):

    y_out_rel_lhand = y_out[:,:,25:40,:] - y_out[:,:,20:21,:]
    y_out_rel_rhand = y_out[:,:,40:,:] - y_out[:,:,21:22,:]

    out_poses_rel_lhand = out_poses[:,:,25:40,:] - out_poses[:,:,20:21,:]
    out_poses_rel_rhand = out_poses[:,:,40:,:] - out_poses[:,:,21:22,:]

    loss_rel_lhand = joint_loss(y_out_rel_lhand,out_poses_rel_lhand)
    loss_rel_rhand = joint_loss(y_out_rel_rhand,out_poses_rel_rhand)

    return loss_rel_lhand + loss_rel_rhand

def joint_body_loss(y_out, out_poses):

    y_out_wrist = y_out[:,:,20:22,:]
    out_poses_wrist = out_poses[:,:,20:22,:]

    y_out_wrist = y_out_wrist.reshape(-1, 3)
    out_poses_wrist = out_poses_wrist.reshape(-1, 3)

    l_wrist = torch.mean(torch.norm(y_out_wrist - out_poses_wrist, 2, 1))

    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    return torch.mean(torch.norm(y_out - out_poses, 2, 1)),l_wrist

def joint_body_loss_test(y_out, out_poses):
    
    y_out_wrist = y_out[:,20:22,:]
    out_poses_wrist = out_poses[:,20:22,:]

    y_out_wrist = y_out_wrist.reshape(-1, 3)
    out_poses_wrist = out_poses_wrist.reshape(-1, 3)

    l_wrist = torch.mean(torch.norm(y_out_wrist - out_poses_wrist, 2, 1))

    y_out = y_out.reshape(-1, 3)
    out_poses = out_poses.reshape(-1, 3)
    return torch.mean(torch.norm(y_out - out_poses, 2, 1)),l_wrist

def bone_length_error(joints, input_bone_lengths, skeleton_cls):
    bone_lengths = calculate_bone_lengths(joints, skeleton_cls)
    return np.sum(np.abs(np.array(input_bone_lengths) - bone_lengths))

def calculate_bone_lengths(joints, skeleton_cls):
    return np.array([np.linalg.norm(joints[bone[0]] - joints[bone[1]] + 0.001) for bone in skeleton_cls.bones])

def bone_loss(raw,predict,device):

	raw_bone_length = cal_bone_loss(raw,device)
	pred_bone_length = cal_bone_loss(predict,device)

	diff = torch.abs(pred_bone_length - raw_bone_length) 
	loss = torch.mean(diff) 

	return loss

def cal_bone_loss(x,device):
    # KCS 
    batch_num = x.size()[0]
    frame_num = x.size()[1]
    joint_num = x.size()[2]

    Ct = get_matrix(device)

    x_ = x.transpose(2, 3) # b, t, 3, 55
    x_ = torch.matmul(x_, Ct)  # b, t, 3, 54
    bone_length = torch.norm(x_, 2, 2) # b, t, 54

    return bone_length

def get_matrix(device,type='all'):

    S_of_lhand = [20, 20, 20, 20, 20, 37, 38, 25, 26, 28, 29, 34, 35, 31, 32]  
    S_of_rhand = [21, 21, 21, 21, 21, 52, 53, 40, 41, 43, 44, 49, 50, 46, 47]  
    S_of_body = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7,   8,  9,  9,  9, 12,   13, 14, 16, 18, 17,  19, 15, 15, 15]  
    E_of_lhand = [37, 25, 28, 34, 31, 38, 39, 26, 27, 29, 30, 35, 36, 32, 33]  
    E_of_rhand = [40, 43, 49, 46, 52, 53, 54, 41, 42, 44, 45, 50, 51, 47, 48]  
    E_of_body = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  11, 12, 13, 14, 15,  16, 17, 18, 20, 19,  21, 22, 23, 24]  

    if type=='all':
        E = np.hstack((E_of_body, E_of_lhand, E_of_rhand))
        S = np.hstack((S_of_body, S_of_lhand, S_of_rhand))
        matrix = torch.zeros([55,54])
    elif type=='lhand':
        E = E_of_lhand
        S = S_of_lhand
        matrix = torch.zeros([55,15])
    elif type=='rhand':
        E = E_of_rhand
        S = S_of_rhand
        matrix = torch.zeros([55,15])
    elif type=='body':
        E = E_of_body
        S = S_of_body
        matrix = torch.zeros([55,24])

    for i in range(S.shape[0]):
        matrix[S[i].tolist(),i] = 1
        matrix[E[i].tolist(),i] = -1
    
    matrix = matrix.to(device)

    return matrix

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
    ckpt_path = f"{opt.ckpt}/{opt.model_type}/ckpt_best.pth.tar"
    logger.info(f"Loading checkpoint from {ckpt_path}")
    
    ckpt = accelerator.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    
    # Load the Accelerator state
    accelerator.load_state(opt.ckpt)
    
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


from __future__ import absolute_import 
from __future__ import print_function  


import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math



class GraphConvolution(nn.Module):
 
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()  
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  
        self.att = Parameter(torch.FloatTensor(node_n, node_n))  
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        support = torch.matmul(x, self.weight)  
        y = torch.matmul(self.att, support) 
        if self.bias is not None:
            return y + self.bias  
        else:
            return y
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):

        super(GC_Block, self).__init__()  
        self.in_features = in_features
        self.out_features = in_features
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)  
        self.do = nn.Dropout(p_dropout)  
        self.act_f = nn.Tanh()  

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)  
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
 
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        for i in range(self.num_stage):
            y = self.gcbs[i](y)
        y = self.gc7(y)
        y = y + x

        return y


from __future__ import absolute_import  
from __future__ import print_function  

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math, copy
from torch.nn import functional as F

from model_others.GCN import *
import util.data_utils as utils

class TempSoftmaxFusion_2(nn.Module):
    def __init__(self, channels, detach_inputs=False, detach_feature=False):
        super(TempSoftmaxFusion_2, self).__init__()
        self.detach_inputs = detach_inputs
        self.detach_feature = detach_feature
        layers = []
        for l in range(0, len(channels) - 1):
            layers.append(nn.Linear(channels[l], channels[l+1]))
            if l < len(channels) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.register_parameter('temperature', nn.Parameter(torch.ones(1)))

    def forward(self, x, y, work=True):
        b, n, f = x.shape
        x = x.reshape(-1, f)
        y = y.reshape(-1, f)
        f_in = torch.cat([x, y], dim=1)
        if self.detach_inputs:
            f_in = f_in.detach()
        f_temp = self.layers(f_in)
        f_weight = F.softmax(f_temp*self.temperature, dim=1)
        if self.detach_feature:
            x = x.detach()
            y = y.detach()
        f_out = f_weight[:,[0]]*x + f_weight[:,[1]]*y
        f_out = f_out.view(b,-1,f)
        return f_out

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def get_mmdloss(source, target,kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, pde_qk, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        query = query.transpose(0,1)
        key = key.transpose(0,1)
        value = value.transpose(0,1)

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
                
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn = attn.transpose(0,1)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class SelfAttention_block(nn.Module):
    def __init__(self, input_feature=60, hidden_feature=256, p_dropout=0.5, num_stage=12, node_n=15*3):
        super(SelfAttention_block, self).__init__()
        self.gcn = GCN(input_feature, hidden_feature, p_dropout, num_stage, node_n)

    def forward(self, x):
        y = self.gcn(x)
        return y

class CrossAttention_block(nn.Module):
    def __init__(self,
                 input_dim=60,
                 head_num=3,
                 dim_ffn=256,
                 dropout=0.2,
                 init_fn=utils.normal_init_):
        super(CrossAttention_block, self).__init__()
        self._model_dim = input_dim
        self._dim_ffn = dim_ffn
        self._relu = nn.ReLU()
        self._dropout_layer = nn.Dropout(dropout)
        self.inner_att = MultiheadAttention(input_dim, head_num, attn_dropout=dropout)
        self._linear1 = nn.Linear(self._model_dim, self._dim_ffn)
        self._linear2 = nn.Linear(self._dim_ffn, self._model_dim)
        self._norm2 = nn.LayerNorm(self._model_dim, eps=1e-5)

        utils.weight_init(self._linear1, init_fn_=init_fn)
        utils.weight_init(self._linear2, init_fn_=init_fn)

    def forward(self, x, y, pdm_xy=None):
        query =x
        key = y
        value = y
        attn_output, _ = self.inner_att(
            query,
            key,
            value,
            pdm_xy
        )
        norm_attn_ = self._dropout_layer(attn_output) + query
        norm_attn = self._norm2(norm_attn_)
        output = self._linear1(norm_attn)
        output = self._relu(output)
        output = self._dropout_layer(output)
        output = self._linear2(output)
        output = self._dropout_layer(output) + norm_attn_
        return output

class DA_Norm(nn.Module):
  def __init__(self,num_features):
    super().__init__()

    shape = (1,1,num_features)
    shape2 = (1,1,num_features)

    self.gamma = nn.Parameter(torch.ones(shape))
    self.beta = nn.Parameter(torch.zeros(shape))
    self.gamma2 = nn.Parameter(torch.ones(shape))
    self.beta2 = nn.Parameter(torch.zeros(shape))

    moving_mean = torch.zeros(shape2)
    moving_var = torch.zeros(shape2)
    moving_mean2 = torch.zeros(shape2)
    moving_var2 = torch.zeros(shape2)

    self.register_buffer("moving_mean", moving_mean)  
    self.register_buffer("moving_var", moving_var)  
    self.register_buffer("moving_mean2", moving_mean2)  
    self.register_buffer("moving_var2", moving_var2)
    self.weight = nn.Parameter(torch.zeros(1))

  def forward(self,X, X2):
    if self.moving_mean.device != X.device:
      self.moving_mean = self.moving_mean.to(X.device)
      self.moving_var = self.moving_var.to(X.device)
      self.moving_mean2 = self.moving_mean2.to(X.device)
      self.moving_var2 = self.moving_var2.to(X.device)
    Y, Y2, self.moving_mean, self.moving_var,self.moving_mean2, self.moving_var2 = batch_norm(X,X2,self.gamma,self.beta,self.moving_mean,self.moving_var,self.gamma2,self.beta2,self.moving_mean2,self.moving_var2,self.weight,eps=1e-5,momentum=0.9)

    return Y,Y2

def batch_norm(X, X2,gamma,beta,moving_mean,moving_var,gamma2,beta2,moving_mean2,moving_var2,weight,eps,momentum):

  if not torch.is_grad_enabled():

    X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    X_hat2 = (X2 - moving_mean2) / torch.sqrt(moving_var2 + eps)
  else:
    weight = (F.sigmoid(weight)+1)/2

    mean = X.mean(dim=(0,1),keepdim=True)
    var = ((X - mean)**2).mean(dim=(0,1),keepdim=True)

    mean2 = X2.mean(dim=(0,1),keepdim=True)
    var2 = ((X2 - mean2)**2).mean(dim=(0,1),keepdim=True)

    mean_fa = weight * mean + (1-weight)* mean2
    mean_fb = weight * mean2 + (1-weight)* mean

    var_fa = weight * var + (1-weight)* var2
    var_fb = weight * var2 + (1-weight)* var

    X_hat = (X - mean_fa) / torch.sqrt(var_fa + eps)
    X_hat2 = (X2 - mean_fb) / torch.sqrt(var_fb + eps)

    moving_mean = momentum * moving_mean + (1.0 - momentum) * mean_fa
    moving_var = momentum * moving_var + (1.0 - momentum) * var_fa

    moving_mean2 = momentum * moving_mean2 + (1.0 - momentum) * mean_fb
    moving_var2 = momentum * moving_var2 + (1.0 - momentum) * var_fb

  Y = gamma * X_hat + beta
  Y2 = gamma2 * X_hat2 + beta2

  return Y, Y2, moving_mean.data, moving_var.data,moving_mean2.data, moving_var2.data

class Alignment_block(nn.Module):
    def __init__(self,
                 input_dim=256,
                 head_num=8,
                 dim_ffn=256,
                 dropout=0.2,
                 init_fn=utils.normal_init_,
                 src_len1=None,
                 src_len2=None):
        super(Alignment_block, self).__init__()
        self._model_dim = input_dim
        self._dim_ffn = dim_ffn

    def forward(self, x, x2, x3, mmd_flag=False):
        # Calculating MMD loss
        output_sa_x = x
        output_sa_x2 = x2
        output_sa_x3 = x3
        if mmd_flag:
            xa_f = torch.mean(output_sa_x,1)
            xb_f = torch.mean(output_sa_x2,1)
            xc_f = torch.mean(output_sa_x3,1)
            mmdlossab = get_mmdloss(xa_f,xb_f)
            mmdlossbc = get_mmdloss(xb_f,xc_f)
            mmdlossac = get_mmdloss(xc_f,xa_f)
        else:
            mmdlossab = 0
            mmdlossbc = 0 
            mmdlossac = 0

        return  output_sa_x, output_sa_x2, output_sa_x3, mmdlossab, mmdlossbc, mmdlossac

class GCN_EAI(nn.Module):
    def __init__(self, input_feature=60, hidden_feature=256, p_dropout=0.5, num_stage=12, lh_node_n=15*3, rh_node_n=15*3,b_node_n=25*3):
        super(GCN_EAI, self).__init__()

        # Individual Encoder
        num_stage_encoder = 12
        self.body_encoder = SelfAttention_block(input_feature, hidden_feature, p_dropout, num_stage_encoder, node_n=b_node_n)
        self.lhand_encoder = SelfAttention_block(input_feature, hidden_feature, p_dropout, num_stage_encoder, node_n=lh_node_n+3)
        self.rhand_encoder = SelfAttention_block(input_feature, hidden_feature, p_dropout, num_stage_encoder, node_n=rh_node_n+3)

        # Distribution Norm
        self._normab = DA_Norm(input_feature)
        self._normbc = DA_Norm(input_feature)
        self._normca = DA_Norm(input_feature)

        # Feature Alignment
        self.align_num_layers = 1
        head_num = 3
        self._align_layers = nn.ModuleList([])
        for i in range(self.align_num_layers):
            self._align_layers.append(Alignment_block(head_num=head_num,input_dim=input_feature,src_len1=b_node_n,src_len2=lh_node_n+3))

        # Semantic Interaction
        self.ca_num_layers = 5
        self._inter_body_lhand_layers = nn.ModuleList([])
        self._inter_body_rhand_layers = nn.ModuleList([])
        self._inter_lhand_body_layers = nn.ModuleList([])
        self._inter_lhand_rhand_layers = nn.ModuleList([])
        self._inter_rhand_lhand_layers = nn.ModuleList([])
        self._inter_rhand_body_layers = nn.ModuleList([])
        for i in range(self.ca_num_layers):
            self._inter_body_lhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_body_rhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_lhand_body_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_lhand_rhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_rhand_lhand_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))
            self._inter_rhand_body_layers.append(CrossAttention_block(head_num=head_num,input_dim=input_feature))

        # Physical Interaction
        self.fusion_lwrist = TempSoftmaxFusion_2(channels=[input_feature*6,input_feature,2])
        self.fusion_rwrist = TempSoftmaxFusion_2(channels=[input_feature*6,input_feature,2])

        # Decoder
        self.body_decoder = nn.Linear(input_feature*3, input_feature)
        self.lhand_decoder = nn.Linear(input_feature*3, input_feature)
        self.rhand_decoder = nn.Linear(input_feature*3, input_feature)
        self.rwrist_decoder = nn.Linear(input_feature*3, input_feature)
        self.lwrist_decoder = nn.Linear(input_feature*3, input_feature)
        utils.weight_init(self.body_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.lhand_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.rhand_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.rwrist_decoder, init_fn_= utils.normal_init_)
        utils.weight_init(self.lwrist_decoder, init_fn_= utils.normal_init_)

    def forward(self, x, action=None, pde_ml=None,pde_lm=None,pde_mr=None,pde_rm=None,pde_lr=None,pde_rl=None):

        # data process & wrist replicate 
        b, n, f = x.shape
        whole_body_x = x.view(b, -1, 3, f)
        lwrist = whole_body_x[:,20:21].detach()
        rwrist = whole_body_x[:,21:22].detach()
        b_x = whole_body_x[:,:25].view(b, -1, f)
        lh_x = torch.cat((lwrist,whole_body_x[:,25:40]),1)
        lh_x = lh_x.view(b, -1, f)
        rh_x = torch.cat((rwrist,whole_body_x[:,40:]),1)
        rh_x = rh_x.view(b, -1, f)

        # Encoding
        hbody = self.body_encoder(b_x)
        lhand = self.lhand_encoder(lh_x)
        rhand = self.rhand_encoder(rh_x)

        # Distribution Normalization
        hbody1,lhand1 = self._normab(hbody,lhand)
        lhand1,rhand1 = self._normbc(lhand1,rhand)
        rhand1,hbody1 = self._normca(rhand1,hbody1)

        # Feature Alignment
        hbody2, rhand2, lhand2, mmdloss_ab, mmdloss_ac, mmdloss_bc = self._align_layers[0](hbody1, rhand1,lhand1,mmd_flag = True)

        # Semantic Interaction
        rhand_2_hbody = hbody2
        lhand_2_hbody = hbody2
        lhand_2_rhand = rhand2
        hbody_2_rhand = rhand2
        rhand_2_lhand = lhand2
        hbody_2_lhand = lhand2

        for i in range(self.ca_num_layers):
            rhand_2_hbody = self._inter_body_rhand_layers[i](rhand_2_hbody, rhand2)
            lhand_2_hbody = self._inter_body_lhand_layers[i](lhand_2_hbody, lhand2)

            lhand_2_rhand = self._inter_rhand_lhand_layers[i](lhand_2_rhand, lhand2)
            hbody_2_rhand = self._inter_rhand_body_layers[i](hbody_2_rhand, hbody2)

            rhand_2_lhand = self._inter_lhand_rhand_layers[i](rhand_2_lhand, rhand2)
            hbody_2_lhand = self._inter_lhand_body_layers[i](hbody_2_lhand, hbody2)

        # Feature Concat
        fusion_body = torch.cat((hbody,rhand_2_hbody,lhand_2_hbody),dim=2)
        fusion_rhand = torch.cat((rhand,lhand_2_rhand,hbody_2_rhand),dim=2)
        fusion_lhand = torch.cat((lhand,rhand_2_lhand,hbody_2_lhand),dim=2)

        # Physical Interaction
        b, n, f1 = fusion_body.shape
        hbody_lwrist = fusion_body.view(b, -1, 3, f1)[:,20:21].view(b, -1, f1)
        hbody_rwrist = fusion_body.view(b, -1, 3, f1)[:,21:22].view(b, -1, f1)
        lhand_lwrist = fusion_lhand.view(b, -1, 3, f1)[:,:1].view(b, -1, f1)        
        rhand_rwrist = fusion_rhand.view(b, -1, 3, f1)[:,:1].view(b, -1, f1)
        fusion_lwrist = self.fusion_lwrist(hbody_lwrist,lhand_lwrist)
        fusion_rwrist = self.fusion_rwrist(hbody_rwrist,rhand_rwrist)

        hbody_no_wrist = torch.cat((fusion_body.view(b, -1, 3, f1)[:,:20],fusion_body.view(b, -1, 3, f1)[:,22:]),1).view(b, -1, f1)
        lhand_no_wrist = fusion_lhand.view(b, -1, 3, f1)[:,1:].view(b, -1, f1)
        rhand_no_wrist = fusion_rhand.view(b, -1, 3, f1)[:,1:].view(b, -1, f1)

        # Decoding
        hbody_no_wrist = self.body_decoder(hbody_no_wrist)
        lhand_no_wrist = self.lhand_decoder(lhand_no_wrist) 
        rhand_no_wrist = self.rhand_decoder(rhand_no_wrist)
        fusion_lwrist = self.lwrist_decoder(fusion_lwrist)
        fusion_rwrist = self.rwrist_decoder(fusion_rwrist)

        hbody_no_wrist = hbody_no_wrist.view(b, -1, 3, f)
        lhand_no_wrist = lhand_no_wrist.view(b, -1, 3, f)
        rhand_no_wrist = rhand_no_wrist.view(b, -1, 3, f)
        fusion_lwrist = fusion_lwrist.view(b, -1, 3, f)
        fusion_rwrist = fusion_rwrist.view(b, -1, 3, f)
        output = torch.cat([hbody_no_wrist[:,:20],fusion_lwrist,fusion_rwrist,hbody_no_wrist[:,20:],lhand_no_wrist,rhand_no_wrist],1).view(b, -1, f) + x

        return output, mmdloss_ab, mmdloss_ac, mmdloss_bc

# coding=utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@IDE: PyCharm
@author: dpx
@contact: dingpx2015@gmail.com
@time: 2022,9月
Copyright (c), xiaohongshu

@Desc:

"""
from multiprocessing.util import is_exiting
import os
import pdb
import numpy
import torch
import shutil
import random
import torch.optim
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from util import loss_func
from util.opt import Options
from util.grab import Grab
from torch.autograd import Variable
from util import utils_utils as utils
from model_others.EAI import GCN_EAI
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



def main(opt, rank, local_rank, world_size, device):

    # 初始化参数
    setup_seed(opt.seed)
    input_n = opt.input_n
    output_n = opt.output_n
    all_n = input_n + output_n
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr

    # 加载数据集
    print(">>> loading train_data")
    train_dataset = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=0, debug= opt.is_debug, using_saved_file=opt.is_using_saved_file, using_noTpose2=opt.is_using_noTpose2)
    print(">>> loading val_data")
    val_dataset   = Grab(path_to_data=opt.grab_data_dict, input_n=input_n, output_n=output_n, split=1, debug= opt.is_debug, using_saved_file=opt.is_using_saved_file,using_noTpose2=opt.is_using_noTpose2)
    print(">>> making dataloader")

    # 多GPU分布式训练的数据处理
    batch_size = opt.train_batch // world_size  # [*] // world_size
    train_sampler = DistributedSampler(train_dataset, shuffle=True)  # [*]
    val_sampler = DistributedSampler(val_dataset, shuffle=False)  # [*]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)  # [*] sampler=...
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)  # [*] sampler=...
    print(">>> train data {}".format(train_dataset.__len__()))  
    print(">>> validation data {}".format(val_dataset.__len__()))  

    # 加载模型
    print(">>> creating model")
    model = GCN_EAI(input_feature=all_n, hidden_feature=opt.linear_size, p_dropout=opt.dropout, num_stage=opt.num_stage, lh_node_n=opt.num_lh*3, rh_node_n=opt.num_rh*3,b_node_n=opt.num_body*3) 
    model_name = '{}'.format(opt.model_type)
    if opt.is_exp:
        ckpt = opt.ckpt + opt.exp
    else:
        ckpt = opt.ckpt + model_name
    
    # 将模型迁移到GPU上
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        if_find_unused_parameters = False
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=if_find_unused_parameters)  # [*] DDP(...)
    print_only_rank0(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 加载优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr*world_size)

    # continue from checkpoint
    script_name = "eai_dct_n{:d}_out{:d}_dctn{:d}".format(input_n, output_n, all_n)
    print_only_rank0(">>> is_load {}".format(opt.is_load))
    if opt.is_load:
        model_path_len = '{}/ckpt_{}_best.pth.tar'.format(ckpt, script_name)
        print_only_rank0(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt_model = torch.load(model_path_len)
        else:
            ckpt_model = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt_model['epoch']
        err_best = ckpt_model['train_loss']
        lr_now = ckpt_model['lr']
        model.load_state_dict(ckpt_model['state_dict'])
        optimizer.load_state_dict(ckpt_model['optimizer'])
        print_only_rank0(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    else:
        print_only_rank0(">>> loading ckpt from scratch")
        # 新建/覆盖ckpt文件
        if dist.get_rank() == 0:
            if os.path.exists(ckpt):
                shutil.rmtree(ckpt)
            os.makedirs(ckpt,exist_ok=True)

    # start training
    print(">>> err_best", err_best)

    dct_trans_funcs = {
        'Norm': get_dct_norm,
        'No_Norm': get_dct,
    }
    idct_trans_funcs = {
        'Norm': get_idct_norm,
        'No_Norm': get_idct,
    }

    # flag设定：是否要对手部做norm
    print('>>> whether hand norm:{}'.format(opt.is_hand_norm))
    if opt.is_hand_norm:
        dct_trans = dct_trans_funcs['Norm']
        idct_trans = idct_trans_funcs['Norm']
    else:
        dct_trans = dct_trans_funcs['No_Norm']
        idct_trans = idct_trans_funcs['No_Norm']

    # 训练
    for epoch in range(start_epoch, opt.epochs):

        # sampler重采样dataloader
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # 学习率衰减设置
        if (epoch + 1) % opt.lr_decay == 0:  # lr_decay=2学习率延迟
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)  # lr_gamma学习率更新倍数0.96
        print_only_rank0('=====================================')
        print_only_rank0('>>> epoch: {} | lr: {:.6f}'.format(epoch + 1, lr_now))

        # csv初始化设置
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])

        # 训练
        Ir_now, t_l, = train(train_loader, model, optimizer, device=device, lr_now=lr_now, max_norm=opt.max_norm,dct_trans=dct_trans,idct_trans=idct_trans,is_boneloss=opt.is_boneloss,is_weighted_jointloss=opt.is_weighted_jointloss)
        # 训练结果
        print_only_rank0("train_loss:{}".format(t_l))
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        # 验证
        v_loss = validate(val_loader, model, device=device,dct_trans=dct_trans,idct_trans=idct_trans)
        # 短时结果
        print_only_rank0("v_loss:{}".format(v_loss))
        ret_log = np.append(ret_log, [v_loss])
        head = np.append(head, ['v_loss'])

        ########################################################################################################################
        # 以下是短时的ckpt保存的代码
        if not np.isnan(v_loss):  # 判断空值 只有数组数值运算时可使用如果v_e不是空值
            is_best = v_loss < err_best  # err_best=10000
            err_best = min(v_loss, err_best)
        else:
            is_best = Falsecd
        ret_log = np.append(ret_log, is_best)  # 内容
        head = np.append(head, ['is_best'])  # 表头
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        if epoch == start_epoch:
            df.to_csv(ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        file_name = ['ckpt_' + script_name + '_epoch_{}.pth.tar'.format(epoch+1), 'ckpt_']

        if dist.get_rank() == 0:
            file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
            utils.save_ckpt({'epoch': epoch + 1,
                            'lr': lr_now,
                            'train_loss': t_l,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                        ckpt_path=ckpt,
                        is_best=is_best,
                        file_name=file_name)   
    

def train(train_loader, model, optimizer, device, lr_now, max_norm, dct_trans, idct_trans, is_boneloss,is_weighted_jointloss):
    print_only_rank0("进入train")
    # 初始化
    iter_num = 0
    t_l = utils.AccumLoss()
    model.train()

    for (input_pose, target_pose) in tqdm(train_loader):
        # 加载数据
        model_input = dct_trans(input_pose)
        n = input_pose.shape[0]  # 16
        if torch.cuda.is_available():
            model_input = model_input.to(device).float()
            target_pose = target_pose.to(device).float()
        
        # 前向传播过程
        out_pose,  mmdloss_ab, mmdloss_ac, mmdloss_bc = model(model_input)
        pred_3d, targ_3d = idct_trans(y_out=out_pose, out_joints=target_pose, device=device)

        # loss计算
        if is_weighted_jointloss:
            loss_jt = loss_func.weighted_joint_loss(pred_3d, targ_3d, ratio=0.6)
        else:
            loss_jt = loss_func.joint_loss(pred_3d, targ_3d)
            loss_pjt = loss_func.relative_hand_loss(pred_3d, targ_3d)
    
        if is_boneloss:
            loss_bl = loss_func.bone_loss(pred_3d, targ_3d, device)
            loss = loss_jt + 0.1 * loss_bl + 0.1 * loss_pjt
        else:
            loss = loss_jt 
        loss = loss + 0.001 * (mmdloss_ab+mmdloss_ac+mmdloss_bc)

        # 反向传播过程
        optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0.
        loss.backward()
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()  # 则可以用所有Variable的grad成员和lr的数值自动更新Variable的数值

        # 更新总体loss结果
        t_l.update(loss.cpu().data.numpy() * n, n)

    return lr_now, t_l.avg

def validate(val_loader, model, device, dct_trans, idct_trans):
    print_only_rank0("进入val")
    # 初始化
    t_l = utils.AccumLoss()
    model.eval()

    for i, (input_pose, target_pose) in enumerate(val_loader):
        
        # 加载数据
        model_input = dct_trans(input_pose)
        n = input_pose.shape[0]  # 64
        if torch.cuda.is_available():
            model_input = model_input.to(device).float()
            target_pose = target_pose.to(device).float()

        # 前向传播过程
        out_pose,  _, _, _ = model(model_input)

        # DCT 转 3D结果
        pred_3d, targ_3d = idct_trans(y_out=out_pose, out_joints=target_pose, device=device)
        
        # 短时的ckpt挑选
        pred_3d = pred_3d
        targ_3d = targ_3d
        loss= loss_func.joint_loss(pred_3d, targ_3d)
        t_l.update(loss.cpu().data.numpy() * n, n)



    return t_l.avg

# 一维DCT变换
def get_dct_matrix(N):
    dct_m = np.eye(N)  # 返回one-hot数组
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)  # 2/35开更
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)  # 矩阵求逆
    return dct_m, idct_m

# 设定种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # 会降低训练速度
     torch.backends.cudnn.deterministic = True

# 多GPU分布式训练的初始化
def setup_DDP(backend="nccl", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    device = torch.device("cuda:{}".format(local_rank))
    if verbose:
        print(f"local rank: {local_rank}, global rank: {rank}, world size: {world_size}")
    return rank, local_rank, world_size, device

# 多GPU分布式训练时候只打印第0个GPU的结果
def print_only_rank0(log):
    if dist.get_rank() == 0:
        print(log)

# 相对plevis的坐标系下：3D转DCT
def get_dct(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints)
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

# 相对plevis的坐标系下：DCT转3D
def get_idct(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).float().to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d

# 身体的关节，相对plevis的坐标系下：3D转DCT； 针对手部，相对wrist关节的坐标系下：3D转DCT；
def get_dct_norm(out_joints):
    batch, frame, node, dim = out_joints.data.shape
    out_joints[:,:,25:40,:] =  out_joints[:,:,25:40,:] - out_joints[:,:,20:21,:]
    out_joints[:,:,40:,:] = out_joints[:,:,40:,:] - out_joints[:,:,21:22,:]
    dct_m_in, _ = get_dct_matrix(frame)
    input_joints = out_joints.transpose(0, 1).reshape(frame, -1).contiguous()
    input_dct_seq = np.matmul((dct_m_in[0:frame, :]), input_joints)
    input_dct_seq = torch.as_tensor(input_dct_seq)
    input_joints = input_dct_seq.reshape(frame, batch, -1).permute(1, 2, 0).contiguous()
    return input_joints

# 身体的关节，相对plevis的坐标系下：DCT转3D； 针对手部，相对wrist关节的坐标系下：DCT转3D；
def get_idct_norm(y_out, out_joints, device):
    batch, frame, node, dim = out_joints.data.shape
    _, idct_m = get_dct_matrix(frame)
    idct_m = torch.from_numpy(idct_m).float().to(device)
    outputs_t = y_out.view(-1, frame).transpose(1, 0)
    outputs_p3d = torch.matmul(idct_m[:, 0:frame], outputs_t)
    outputs_p3d = outputs_p3d.reshape(frame, batch, -1, dim).contiguous().transpose(0, 1)
    outputs_p3d[:,:,25:40,:] = outputs_p3d[:,:,25:40,:] + outputs_p3d[:,:,20:21,:]
    outputs_p3d[:,:,40:,:] = outputs_p3d[:,:,40:,:] + outputs_p3d[:,:,21:22,:]
    pred_3d = outputs_p3d
    targ_3d = out_joints
    return pred_3d, targ_3d


if __name__ == "__main__":
    option = Options().parse()
    # 初始化ddp的代码
    rank, local_rank, world_size, device = setup_DDP(verbose=True)
    main(option, rank, local_rank, world_size, device)

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

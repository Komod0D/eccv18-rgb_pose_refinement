import time
import cupy
import cudnn
import torch
import numpy as np
import cv2
from pytorch3d.transforms import quaternion_invert, quaternion_apply, quaternion_to_matrix
from torchvision import transforms

from scipy.spatial.transform import Rotation as R
import os
import json
from render_swisscube import to_homo, Renderer
from rendering.utils import *
from models.MyInception import MyInception

MINIBATCH_SIZE = 16
r = Renderer(synthetic=True)
intrinsic = torch.from_numpy(r.intrinsic).cuda().detach()
height, width = 480, 640
c_height, c_width = 299, 299
n_samples = 256


def crop_middle(img, x, y):
    return img[y - c_height / 2: y + c_height / 2, x - c_width / 2, x + c_width / 2].copy()


def load_network(path=None):
    network = MyInception().cuda(device=0)

    if path is not None:
        weights = torch.load(path)
        network.load_state_dict(weights)
    network = torch.nn.DataParallel(network, device_ids=[0]).cuda(device=0)
    cudnn.benchmark = True
    network.train()

    return network


def my_loss(dq, dt, ds, vs, cam):
    inside = quaternion_apply(dq.unsqueeze(1), vs) + dt.unsqueeze(1)  # N, 256, 3

    project = torch.zeros_like(inside[:, :, :2]).cuda()  # N, 256, 2
    project[:, :, 0] = cam[0, 2] + inside[:, :, 0] * cam[0, 0] / inside[:, :, 2]
    project[:, :, 1] = cam[1, 2] + inside[:, :, 1] * cam[1, 1] / inside[:, :, 2]

    project = project.unsqueeze(1)  # N, 1, 256, 2

    project[:, 0] = (2 * project[:, 0] - width) / width
    project[:, 1] = (2 * project[:, 1] - height) / height  # between -1 and 1

    ds = ds.unsqueeze(1)   # N, 1, H, W

    return torch.sum(torch.nn.functional.grid_sample(ds, project))


def compute_errors(dr, dt, poses_src, poses_tgt):
    dtotal = np.zeros(shape=(dr.shape[0], 4, 4))
    dtotal[:, 3, 3] = 1
    dtotal[:, :3, :3] = dr
    dtotal[:, :3, 3] = dt

    poses_est = dtotal @ poses_src
    rs_est = poses_est[:, :3, :3]
    ts_est = poses_est[:, :3, 3]

    errors_r = [r_true @ np.linalg.inv(r_est) for r_true, r_est in zip(poses_tgt[:, :3, :3], rs_est)]
    error_r = np.mean([np.sum(R.from_matrix(r).as_euler('xyz')) for r in errors_r])  # sum of sum of euler angles

    error_t = np.mean([np.linalg.norm(t) for t in poses_tgt[:, :3, 3] - ts_est])

    return error_r, error_t


def generate_samples(split='training'):
    base_path = '/cvlabdata2/cvlab/datasets_protopap/swisscube/'

    images_list_path = os.path.join(base_path, f'{split}.txt')
    with open(images_list_path, 'r') as f:
        images_list = f.readlines()

    images_list = [images_list[i] for i in np.random.permutation(len(images_list))]

    images = torch.FloatTensor(MINIBATCH_SIZE, 3, c_height, c_width).cuda().detach()
    renders = torch.FloatTensor(MINIBATCH_SIZE, 3, c_height, c_width).cuda().detach()
    ds = torch.FloatTensor(MINIBATCH_SIZE, c_height, c_width).cuda().detach()
    vs = torch.FloatTensor(MINIBATCH_SIZE, n_samples, 3).cuda().detach()

    poses_src = np.zeros((MINIBATCH_SIZE, 4, 4), dtype=np.float32)
    poses_tgt = np.zeros((MINIBATCH_SIZE, 4, 4), dtype=np.float32)
    idx = 0
    for img_path in images_list:
        full_path = os.path.join('/cvlabdata2/home/yhu/data/SwissCube_1.0', img_path.strip())
        num = str(int(os.path.splitext(os.path.basename(full_path))[0]))

        seq_name = os.path.dirname(os.path.dirname(full_path))

        poses_name = os.path.join(seq_name, 'scene_gt.json')
        with open(poses_name, 'r') as j:
            poses = json.load(j)

        pose_dict = poses[num][0]

        rotation = np.array(pose_dict['cam_R_m2c']).reshape((3, 3))
        rotation_q = R.from_matrix(rotation).as_quat()
        translation = np.array(pose_dict['cam_t_m2c'])
        pose_tgt = to_homo(rotation, translation)

        rnd = np.random.uniform(0, 1)
        if rnd > 0.5:
            pose_src = perturb_pose(pose_tgt, 0.1, 10)
        else:
            pose_src = perturb_pose(pose_tgt, 0.01, 100)

        poses_tgt[idx] = pose_tgt

        r.set_pose(np.concatenate(rotation_q, translation))
        _, depth_tgt = r.render_()


        im_real = cv2.imread(full_path)
        im_real = cv2.resize(im_real, (width, width), cv2.INTER_AREA)
        im_real = im_real[width / 2 - height / 2:width / 2 + height / 2]
        im_real = cv2.cvtColor(im_real, cv2.COLOR_BGR2RGB)

        rotation_q, translation = R.from_matrix(pose_src[:3, :3]).as_quat(), pose_src[:3, 3]
        r.set_pose(np.concatenate(rotation_q, translation))
        render_src, depth_src = r.render_()

        images[idx] = preprocess_img(np.transpose(im_real, (2, 0, 1))).cuda().detach() / 255  # 3, H, W
        renders[idx] = preprocess_img(np.transpose(render_src, (2, 0, 1))).cuda().detach() / 255  # 3, H, W

        vs[idx] = get_viewpoint_cloud(depth_src, r.intrinsic, n_samples)  # 256, 3
        ds[idx] = preprocess_depth(distance_transform(depth_tgt)[np.newaxis, :]).cuda().detach()  # 1, H, W


        if idx == MINIBATCH_SIZE:
            yield images, renders, ds, vs, poses_src, poses_tgt
            idx = 0
        else:
            idx += 1


def train(network, optimizer):
    n_batches = 0
    total_loss = 0
    start = time.time()
    for batch_idx, sample in enumerate(generate_samples('training')):
        images, renders, ds, vs, poses_src, poses_tgt = sample
        dq, dt = network(images, renders)

        loss = my_loss(dq, dt, ds, vs, intrinsic)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_batches += 1

        dr = quaternion_to_matrix(dq).cpu().numpy()
        dt = dt.cpu().numpy()
        lr, lt = compute_errors(dr, dt, poses_src, poses_tgt)

        print(f'loss {loss.item():.6f}, error (r {lr:.4f}, t {lt:.4f})'
              f'time: {time.time() - start:.2f}')
        start = time.time()

    return total_loss / n_batches


preprocess_img = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

preprocess_depth = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor()
])

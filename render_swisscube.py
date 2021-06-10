import pyrender
import os
import trimesh
import numpy as np
import cv2
import os
import torch

from scipy.spatial.transform import Rotation as R

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def to_homo(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform


def get_corners(mesh, intrinsic, rotation, translation):
    box = mesh.bounding_box.to_mesh()
    vertices = np.array(box.vertices)
    proj = intrinsic @ (rotation @ vertices.T + translation[:, np.newaxis])
    proj[0] = proj[0] / proj[2]
    proj[1] = proj[1] / proj[2]

    return proj[:2].T


def add_pose_contour(mesh, intrinsic, rotation, translation, color, image, img_scaling=4, thickness=1):
    image = np.copy(image)
    height, width, _ = image.shape
    vs = get_corners(mesh, intrinsic, rotation, translation) / img_scaling
    ps = [(int(vs[i, 0]), int(vs[i, 1])) for i in range(vs.shape[0])]

    # z direction
    for i in range(4):
        cv2.line(image, ps[2 * i], ps[2 * i + 1], color, thickness=thickness)

    # y direction
    for j in range(2):
        for i in range(2):
            cv2.line(image, ps[i + 4 * j], ps[i + 2 + 4 * j], color, thickness=thickness)

    # x direction
    for i in range(4):
        cv2.line(image, ps[i], ps[i + 4], color, thickness=thickness)

    return image


def quaternion2rotation(quat):
        '''
        Do not use the quat2dcm() function in the SPEED utils.py, it is not rotation
        '''
        assert (len(quat) == 4)
        # normalize first
        quat = quat / np.linalg.norm(quat)
        a, b, c, d = quat

        a2 = a * a
        b2 = b * b
        c2 = c * c
        d2 = d * d
        ab = a * b
        ac = a * c
        ad = a * d
        bc = b * c
        bd = b * d
        cd = c * d

        # s = a2 + b2 + c2 + d2

        m0 = a2 + b2 - c2 - d2
        m1 = 2 * (bc - ad)
        m2 = 2 * (bd + ac)
        m3 = 2 * (bc + ad)
        m4 = a2 - b2 + c2 - d2
        m5 = 2 * (cd - ab)
        m6 = 2 * (bd - ac)
        m7 = 2 * (cd + ab)
        m8 = a2 - b2 - c2 + d2

        return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)


class Renderer:
    def __init__(self, synthetic=False):
        self.synthetic = synthetic
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        tscene = trimesh.load('/cvlabdata2/cvlab/datasets_protopap/deepim/data/models/swisscube/swisscube.obj')
        mesh = pyrender.Mesh.from_trimesh(list(tscene.geometry.values()), smooth=False)

        if synthetic:
            width, height = 1024, 1024
        else:
            width, height = 2048, 2048


        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)
        scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[0, 0, 0])

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1000000.0)
        
        if synthetic:
            fx, fy, cx, cy = 607, 607, 512, 512
        else:
            fx, fy, cx, cy = 4000, 4000, 1024, 1024

        self.intrinsic = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape((3, 3))
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, zfar=2000)
        cam_rot = R.from_euler('y', 180, degrees=True).as_matrix()
        cam_matrix = to_homo(cam_rot, np.zeros((3,)))

        self.nm = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        nl = pyrender.Node(light=light, matrix=np.eye(4))
        nc = pyrender.Node(camera=cam, matrix=cam_matrix)

        scene.add_node(self.nm)
        scene.add_node(nl)
        scene.add_node(nc)

        self.scene = scene

    def set_light_pos(self, *args):
        pass

    def set_light_color(self, *args):
        pass

    def set_projection_matrix(self, width, height, fx, fy, px, py, znear, zfar):
        pass

    def set_poses(self, poses):
        self.set_pose(poses[0])

    def set_pose(self, pose):
        pose = np.array(pose)
        if pose.shape[0] == 9:
            pose = pose[2:]
        translation, rotation_quat = pose[4:], pose[:4]
        translation = np.array(translation)
        if not self.synthetic:
            rotation = quaternion2rotation(rotation_quat)
        else:
            rotation = R.from_quat(rotation_quat).as_matrix()
            rotation = rotation @ R.from_euler('x', 90, degrees=True).as_matrix()

        transform = to_homo(rotation, translation)
        self.scene.set_pose(self.nm, pose=transform)

    def render_(self):
        color, depth = self.renderer.render(self.scene)
        color = cv2.resize(color, (640, 640), cv2.INTER_AREA)
        color = color[80:560]
        
        depth = cv2.resize(depth, (640, 640), cv2.INTER_AREA)
        depth = depth[80:560]
        return np.flip(color, (0, 1)).copy(), np.flip(depth, (0, 1)).copy()

    def render(self, image_tensor):
        rgb = self.render_()

        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        image_tensor.copy_(tensor)


    def get_next(self, iteritems):
         
        if not self.synthetic:
            img, pose = next(iteritems)
            img = os.path.join(img.split('/')[0], 'Test', img.split('/')[1])
            img = cv2.imread(img)
            img = cv2.resize(img, (640, 640), cv2.INTER_AREA)
            img = img[80:560]
            rotation = pose['rotation_m2c']
            translation = pose['translation_m2c']
            
        else:
            img_path = next(iteritems).strip()
            full_path = os.path.join('/cvlabdata2/home/yhu/data/SwissCube_1.0', img_path)
            num = str(int(os.path.splitext(os.path.basename(full_path))[0]))
            img = cv2.imread(full_path)
            img = cv2.resize(img, (640, 640), cv2.INTER_AREA)
            img = img[80:560]
            seq_name = os.path.dirname(os.path.dirname(full_path))
            
            poses_name = os.path.join(seq_name, 'scene_gt.json')
            with open(poses_name, 'r') as j:
                poses = json.load(j)

            
            pose = poses[num][0]
            translation = np.array(pose['cam_t_m2c'])
            rotation = np.array(pose['cam_R_m2c']).reshape((3, 3))
            rotation = R.from_mat(rotation).as_matrix()
            

        return img, translation, rotation


if __name__ == '__main__':
    r = Renderer(True)

    import os
    import json
    
    if r.synthetic:
        os.chdir('/cvlabdata2/home/yhu/data/SwissCube_1.0')
        with open('training.txt', 'r') as f:
            images = f.readlines()

        iteritems = iter(images)
        
    else:
        os.chdir('/cvlabdata2/cvlab/datasets_protopap/SwissCubeReal')
        with open('data.json', 'r') as f:
            poses = json.load(f)

        iteritems = iter(poses.items())
    
    img, translation, rotation = r.get_next(iteritems)
    translation = np.array(translation)
    cv2.imshow('image', img)
    x, y, z = translation
    dx, dy, dz = 0, 0, 0
    while True:
        rotation = R.from_quat(rotation).as_matrix() @ R.from_euler('xyz', [dx, dy, dz], degrees=True).as_matrix()
        rotation = R.from_matrix(rotation).as_quat()
        a, b, c, d = rotation
        pose = [a, b, c, d, x, y, z]
        r.set_pose(pose)
        color, depth = r.render_()
        print(depth.max())
        cv2.imshow('render', color)
        cv2.imshow('depth', depth)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == 13:
            img, translation, rotation = r.get_next(iteritems)
            x, y, z = translation
            a, b, c, d = rotation
           
            cv2.imshow('image', img)
        elif key == 119:
            y += 10
        elif key == 115:
            y -= 10
        elif key == 97:
            x -= 10
        elif key == 100:
            x += 10
        elif key == 101:
            z += 10
        elif key == 113:
            z -= 10
        elif key == 81:
            dy -= 15
        elif key == 83:
            dy += 15
        elif key == 82:
            dx -= 15
        elif key == 84:
            dx += 15
        elif key == 85:
            dz += 15
        elif key == 86:
            dz -= 15


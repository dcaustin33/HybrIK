"""Batch Image Processing Demo Script."""
import argparse
import os

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--device',
                    help='device to use',
                    default="cpu",
                    type=str)
parser.add_argument('--img-dir',
                    help='image folder',
                    default='',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = './pretrained_models/hybrik_hrnet.pth'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

det_model = fasterrcnn_resnet50_fpn(pretrained=True)

hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

det_model.to(opt.device)
hybrik_model.to(opt.device)
det_model.eval()
hybrik_model.eval()

files = os.listdir(opt.img_dir)
smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

# List to hold all image data and results
batch_images = []
batch_bboxes = []
batch_filenames = []

# Load and preprocess images
for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        if file[:4] == 'res_':
            continue

        img_path = os.path.join(opt.img_dir, file)
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.device)
        det_output = det_model([det_input])[0]
        tight_bbox = get_one_box(det_output)  # xyxy

        pose_input, bbox, img_center = transformation.test_transform(input_image, tight_bbox)
        pose_input = pose_input.to(opt.device)[None, :, :, :]

        batch_images.append(pose_input)
        batch_bboxes.append((bbox, img_center))
        batch_filenames.append(file)

# Process all images in the batch
for i in range(len(batch_images)):
    pose_input = batch_images[i]
    bbox, img_center = batch_bboxes[i]
    file = batch_filenames[i]

    pose_output = hybrik_model(
        pose_input, flip_test=True,
        bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
        img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
    )

    transl = pose_output.transl.detach()
    vertices = pose_output.pred_vertices.detach()
    verts_batch = vertices
    transl_batch = transl

    focal = 1000.0 / 256 * xyxy2xywh(bbox)[2]
    color_batch = render_mesh(
        vertices=verts_batch, faces=smpl_faces,
        translation=transl_batch,
        focal_length=focal, height=input_image.shape[0], width=input_image.shape[1])

    valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
    image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
    image_vis_batch = (image_vis_batch * 255).cpu().numpy()

    color = image_vis_batch[0]
    valid_mask = valid_mask_batch[0].cpu().numpy()
    input_img = input_image
    alpha = 0.9
    image_vis = alpha * color[:, :, :3] * valid_mask + (
        1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

    image_vis = image_vis.astype(np.uint8)
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    res_path = os.path.join(opt.out_dir, file)
    cv2.imwrite(res_path, image_vis)
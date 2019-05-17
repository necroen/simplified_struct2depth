from __future__ import division
import torch
import torch.nn.functional as F


import numpy as np

from params import args


pixel_coords = None



'''
def get_transform_mat(egomotion_vecs, i, j):
  """Returns a transform matrix defining the transform from frame i to j."""
  egomotion_transforms = []
  batchsize = tf.shape(egomotion_vecs)[0]
  if i == j:
    return tf.tile(tf.expand_dims(tf.eye(4, 4), axis=0), [batchsize, 1, 1])

  for k in range(min(i, j), max(i, j)):
    transform_matrix = _egomotion_vec2mat(egomotion_vecs[:, k, :], batchsize)

    if i > j:  # Going back in sequence, need to invert egomotion.
      egomotion_transforms.insert(0, tf.linalg.inv(transform_matrix))
    else:  # Going forward in sequence
      egomotion_transforms.append(transform_matrix)

  # Multiply all matrices.
  egomotion_mat = egomotion_transforms[0]
  for i in range(1, len(egomotion_transforms)):
    egomotion_mat = tf.matmul(egomotion_mat, egomotion_transforms[i])
  return egomotion_mat

'''

def get_transform_mat(egomotion_vecs, i, j):
    egomotion_transforms = []
    batchsize = egomotion_vecs.size()[0] # or batchsize = args.batchsize

    if i == j:
        return torch.eye(4, 4).expand(batchsize, 4, 4)
    for k in range(min(i, j), max(i, j)):
        #                   tf  _egomotion_vec2mat  返回[1, 4, 4]
        transform_matrix = pose_vec2mat(egomotion_vecs[:, k, :], batchsize)

        if i > j:
            egomotion_transforms.insert(0, transform_matrix.inverse() )
        else:
            egomotion_transforms.append(transform_matrix )
    
    egomotion_mat = egomotion_transforms[0]
    for i in range(1, len(egomotion_transforms)):
        egomotion_mat = torch.matmul(egomotion_mat, egomotion_transforms[i]) # [1, 4, 4] x [1, 4, 4] ?
    return egomotion_mat





def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    # if rotation_mode == 'euler':
    rot_mat = euler2mat(rot)  # [B, 3, 3]

    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]

    a = torch.tensor([0, 0, 0, 1]) # 这里有一个涉及device类型和batchsize参数 要改

    # batchsize = args.batchsize # 这样写会报错，因为如果最后一个batch不足batchsize的话
    batchsize = transform_mat.size()[0]
    a = a.expand((batchsize, 1, 4)).float()

    a = a.to(args.device)

    transform_mat = torch.cat( [transform_mat, a], dim=1 )   #  [B, 4, 4]

    return transform_mat


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat





def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 
                                                'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """
    Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """
    Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates 
        system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2] x在前，y在后
    return pixel_coords.reshape(b,h,w,2)



def quat2mat(quat):
    """
    Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. 
        fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2,        2*xy - 2*wz,        2*wy + 2*xz,
                                2*wz + 2*xy,  w2 - x2 + y2 - z2,        2*yz - 2*wx,
                                2*xz - 2*wy,        2*wx + 2*yz,  w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat




def inverse_warp(img, depth, pose_mat, intrinsics, rotation_mode='euler', padding_mode='zeros'):
    # intrin 是 [1, 3, 3]
    # 传进来的pose 是 [1, 4, 4] 的 mat
    """
    warp_img = K*pose*depth*K^(-1)*I_t

    Inverse warp a source image to the target image plane.
    将 t-1  t+1 时刻的source img  warp 到 t 时刻的target img

    这里 B = 2
    传进来的 img 都是 ref_img， 也就是 source img
    
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
#    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam( depth, intrinsics.inverse() )  # [B,3,H,W]

#    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]
    #                        3x3         3 x 4   - > 3 x 4
    # 现在                   3x3        4 x 4  -> ?  把posemat最后一行截掉去

    src_pixel_coords = cam2pixel(cam_coords, 
                                proj_cam_to_src_pixel[:,:,:3], 
                                proj_cam_to_src_pixel[:,:,-1:], 
                                padding_mode)    # [B,H,W,2]  x在前，y在后

    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    
    # _spatial_transformer
    px = src_pixel_coords[:, :, :, :1]
    py = src_pixel_coords[:, :, :, 1:]

    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0

    # _bilinear_sampler  怎么知道torch里头的操作能不能够微分？
    px = torch.reshape(px, (-1,))
    py = torch.reshape(py, (-1,))
    
    px = px.float()
    py = py.float()

    img_height_f = float(img_height)
    img_width_f = float(img_width)

    px = (px + 1.0) * (img_width_f - 1.0) / 2.0
    py = (py + 1.0) * (img_height_f - 1.0) / 2.0

    x1 = px.int() + 1
    y1 = py.int() + 1

    # mask = tf.logical_and(
    #     tf.logical_and(x0 >= zero, x1 <= max_x),
    #     tf.logical_and(y0 >= zero, y1 <= max_y)
    #     )
    # mask = tf.to_float(mask)

    zeros = torch.zeros_like(px)

    r1 = torch.ge(px, zeros)
    r2 = torch.le(x1, (img_width_f-1)*torch.ones_like(x1) )

    r3 = torch.ge(py, zeros)
    r4 = torch.le(y1, (img_height_f-1)*torch.ones_like(y1) )

    mask = (r1 & r2) & (r3 & r4)

    mask = mask.float()

    # mask = torch.reshape(mask, (1, img_height, img_width, 1)) # origin

    mask = torch.reshape(mask, (batch_size, img_height, img_width, 1))

    return projected_img, mask








if __name__ == "__main__":
    egomotion_vecs = torch.rand([1, 2, 6])
    
    rst = get_transform_mat(egomotion_vecs, 0, 2)
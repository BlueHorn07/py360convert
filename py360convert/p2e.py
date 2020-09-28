import math
import cv2
import numpy as np

from . import utils

def p2e(p_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0):
  """
    p_img:  ndarray in shape of [H, W, C] \\
    fov_deg: scalar or (scalar, scalar) field of view in degree \\
    u_deg:      horizon viewing angle in range [-180, 180] \\
    v_deg:      vertical viewing angle in range [-90, 90] \\
    in_rot_deg: fov rotating angle in range [0, 360] \\
  """

  assert len(p_img.shape) == 3
  h, w, channel = p_img.shape
  out_h, out_w = out_hw

  ## check fov_deg is scalar or (sclar, scalar)
  try:
    if(isinstance(fov_deg, (int, float))):
      h_fov, v_fov = math.radians(fov_deg), math.radians(fov_deg)
    else:
      h_fov, v_fov = math.radians(fov_deg[0]), math.radians(fov_deg[1])
  except Exception as e:
    print("incorrect fov_deg", e)
  h_len = np.tan(h_fov / 2)
  v_len = np.tan(v_fov / 2)
  
  u = math.radians(u_deg)
  v = math.radians(v_deg)

  x, y = np.meshgrid(np.linspace(-180, 180, out_w),np.linspace(90,-90, out_h))

  x_map = np.cos(np.radians(y)) * np.cos(np.radians(x))
  y_map = np.cos(np.radians(y)) * np.sin(np.radians(x))
  z_map = np.sin(np.radians(y))

  xyz = np.dstack((x_map, y_map, z_map))

  z_axis = np.array([0.0, 0.0, 1.0], np.float32)
  y_axis = np.array([0.0, 1.0, 0.0], np.float32)
  Ry = utils.rotation_matrix(-v, y_axis)
  Rz = utils.rotation_matrix(u, np.dot(z_axis, Ry))
  xyz = xyz.dot(Ry).dot(Rz)

  inverse_mask = np.where(xyz[:,:,0] > 0, 1, 0)

  xyz[:,:] = xyz[:,:] / np.dstack((xyz[:,:,0], xyz[:,:,0], xyz[:,:,0]))

  lon_map = np.where((-h_len<xyz[:,:,1])&(xyz[:,:,1]<h_len)&(-v_len<xyz[:,:,2])
              &(xyz[:,:,2]<v_len), (xyz[:,:,1]+h_len)/2/h_len*w, 0)
  lat_map = np.where((-h_len<xyz[:,:,1])&(xyz[:,:,1]<h_len)&(-v_len<xyz[:,:,2])
              &(xyz[:,:,2]<v_len),(-xyz[:,:,2]+v_len)/2/v_len*h,0)
  mask = np.where((-h_len<xyz[:,:,1])&(xyz[:,:,1]<h_len)&(-v_len<xyz[:,:,2])
              &(xyz[:,:,2]<v_len), 1, 0)
  
  persp = cv2.remap(p_img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
  mask = mask * inverse_mask
  mask = np.dstack((mask, mask, mask))
  persp = persp * mask ## make black region & remove dual image

  return persp.astype('uint8')

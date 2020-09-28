import math
import numpy as np

from . import utils


def e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear'):
  """
    e_img:      ndarray in shape of [H, W, *] \\
    fov_deg:    scalar or (scalar, scalar) field of view in degree \\
    u_deg:      horizon viewing angle in range [-180, 180] \\
    v_deg:      vertical viewing angle in range [-90, 90] \\
    in_rot_deg: fov rotating angle in range [0, 360] \\
    mode:       interpolation mode
  """
  assert len(e_img.shape) == 3
  h, w, channel = e_img.shape

  ## check fov_deg is scalar or (sclar, scalar)
  try:
      h_fov, v_fov = math.radians(fov_deg[0]), math.radians(fov_deg[1])
  except Exception:
      h_fov, v_fov = math.radians(fov_deg), math.radians(fov_deg)

  in_rot = math.radians(in_rot_deg)

  if mode == 'bilinear':
      order = 1
  elif mode == 'nearest':
      order = 0
  else:
      raise NotImplementedError('unknown mode')

  u = -u_deg * np.pi / 180
  v = v_deg * np.pi / 180

  ## get xyz coordinate of target fov reigon
  xyz = utils.xyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
  # print("in e2p")
  # print(xyz)
  ## get uv coordinate of target fov reigon
  uv = utils.xyz2uv(xyz)

  ## get exact pixel locations of target fov reigon
  coor_xy = utils.uv2coor(uv, h, w)

  ## sample fov with exact pixel coordinate
  pers_img = np.stack([
      utils.sample_equirec(e_img[..., i], coor_xy, order=order)
      for i in range(channel)
  ], axis=-1)

  return pers_img

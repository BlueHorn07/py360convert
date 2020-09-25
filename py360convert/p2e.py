import math
import numpy as np

from . import utils

def p2e(p_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0):
  """
    p_img:  ndarray in shape of [H, W, *] \\
    fov_deg: scalar or (scalar, scalar) field of view in degree \\
    u_deg:      horizon viewing angle in range [-180, 180] \\
    v_deg:      vertical viewing angle in range [-90, 90] \\
    in_rot_deg: fov rotating angle in range [0, 360] \\
  """

  assert len(p_img.shape) == 3
  h, w, channel = p_img.shape

  ## check fov_deg is scalar or (sclar, scalar)
  try:
    if(isinstance(fov_deg, (int, float))):
      h_fov, v_fov = fov_deg, fov_deg
    else:
      h_fov, v_fov = math.radians(fov_deg[0]), math.radians(fov_deg[1])
  except Exception as e:
    print("incorrect fov_deg", e)

  in_rot = math.radians(in_rot_deg)

  u = -u_deg * np.pi / 180
  v = v_deg * np.pi / 180

  xyz = utils.xyzpers(h_fov, v_fov, u, v, (h, w), in_rot)
  # print("in p2e")
  # print(xyz)
  uv = utils.xyz2uv(xyz)
  coor_xy = utils.uv2coor(uv, out_hw[0], out_hw[1]).astype('uint')

  equirec = utils.sample_pers(p_img, coor_xy, out_hw)

  return equirec

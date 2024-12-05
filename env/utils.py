import numpy as np

def predict_pos(now, v, step):
  v = np.array(v)
  if v.shape[0] != 3:
    v = np.array([v[0], 0, v[1]])
  t = np.array(now) + v * step / 12
  if t[0] > 0.5:
    t[0] = 1 - t[0]
  if t[0] < -0.5:
    t[0] = -1 - t[0]
  if t[2] > 0.5:
    t[2] = 1 - t[2]
  if t[2] < 0.1:
    t[2] = 0.2 - t[2]
  return t


# pos1 在 pos2 的哪个方向, 只看x和y坐标
#  o
# / \
# 2  \
#    1
# 1 在 2 的右边(right)
# right = 1, left = -1, center = 0
def relative_dir(pos1:dict[str, float], pos2:dict[str, float], use_int=False) -> str:
    angle1 = np.arctan2(pos1['y'], pos1['x'])
    angle2 = np.arctan2(pos2['y'], pos2['x'])
    if abs(angle1 - angle2) < 0.2:
        return 'center' if not use_int else 0
    elif angle1 - angle2 > 0:
        return 'right' if not use_int else 1
    elif angle1 - angle2 < 0:
        return 'left' if not use_int else -1
    return 'center'


def next_tar_step(now, tar1, max):
  tar = [tar1, (max + tar1) / 2, max]
  for t in tar:
    if t >= now:
      return t - now
  return 0
  
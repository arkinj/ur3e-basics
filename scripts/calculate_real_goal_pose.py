import numpy as np
import math

def main():
    # 1mm T, ccw from upper left
    corner_offsets = np.array([
        [-3, -1],
        [1, 3],
        [2, 2],
        [.5, .5],
        [3.5, -2.5],
        [2.5, -3.5],
        [-.5, -.5],
        [-2, -2],
    ], dtype=float) / np.sqrt(2)
    corner_offsets *= 30
    goal_pos = np.array([0, 350])
    ref_pos = np.array([0, 350])
    corner_offsets += goal_pos - ref_pos

    corner_offsets[:,0] *= -1

    np.set_printoptions(precision=1, suppress=True)
    print(corner_offsets)
    

if __name__=='__main__':
    main()
import numpy as np
from scipy.spatial.transform import Rotation

# input
p_0 = np.array([0.4, 1.5, -160.01])
dx_buildspace = np.array([0.0, 0.0, 29.0])
r_buildspace = np.array([0.0, 20.0, 0.0])

# output
dx_hexapod = np.array([0.0, 0.0, 0.0])
r_hexapod = np.array([0.0, 0.0, 0.0])

# conversion
a, b, c = r_buildspace

R_b = Rotation.from_euler('ZYX', [c, b, a], degrees=True)
R_h = R_b.inv()

r_hexapod = R_h.as_euler('xyz', degrees=True)

dx_hexapod = R_h.apply(p_0) - R_h.apply(dx_buildspace) - p_0

# print results
formatter = {'float_kind':lambda x: f"{x:.2f}"}
position_str = np.array2string(dx_hexapod, separator=', ', formatter=formatter)
rotation_str = np.array2string(r_hexapod, separator=', ', formatter=formatter)
print("Hexapod Position (mm):", position_str)
print("Hexapod Rotation (°):", rotation_str)

# reverse conversion
R_h_reverse = Rotation.from_euler('xyz', r_hexapod, degrees=True)
        
R_b_reverse = R_h_reverse.inv()
        
dx_buildspace_reverse = R_b_reverse.apply(R_h_reverse.apply(p_0) - dx_hexapod - p_0)
        
r_buildspace_reverse_zyz = R_b_reverse.as_euler('ZYX', degrees=True)
r_buildspace_reverse = np.array([r_buildspace_reverse_zyz[2], r_buildspace_reverse_zyz[1], r_buildspace_reverse_zyz[0]])

# print reverse results
position_str_reverse = np.array2string(dx_buildspace_reverse, separator=', ', formatter=formatter)
rotation_str_reverse = np.array2string(r_buildspace_reverse, separator=', ', formatter=formatter)
print("Buildspace Position (mm):", position_str_reverse)
print("Buildspace Rotation (°):", rotation_str_reverse)
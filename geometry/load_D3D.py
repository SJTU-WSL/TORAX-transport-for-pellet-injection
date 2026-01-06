import json
import matplotlib.pyplot as plt

js = json.load(open('../PAM/template_D3D_1layer_2species.json'))

# update (Z, R)
plt.figure(figsize=(6, 4))
z = js['equilibrium']['time_slice'][0]['boundary']['outline']['z']
r = js['equilibrium']['time_slice'][0]['boundary']['outline']['r']

magnet_axis_z = js['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['z']
magnet_axis_r = js['equilibrium']['time_slice'][0]['global_quantities']['magnetic_axis']['r']

geometric_axis_z = js['equilibrium']['time_slice'][0]['profiles_1d']['geometric_axis']['z']
geometric_axis_r = js['equilibrium']['time_slice'][0]['profiles_1d']['geometric_axis']['r']

profiles_2d_z = js['equilibrium']['time_slice'][0]['profiles_2d'][0]['grid']['dim2']
profiles_2d_r = js['equilibrium']['time_slice'][0]['profiles_2d'][0]['grid']['dim1']

# plt.xlim(-0, 4)
# plt.ylim(-2, 2)
plt.scatter(profiles_2d_z, profiles_2d_r)
plt.scatter(geometric_axis_r, geometric_axis_z, color='y')
plt.scatter(magnet_axis_r, magnet_axis_z, color='green')
plt.scatter(r, z, s=2)
plt.grid()
plt.show()
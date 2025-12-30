import json
import re
import matplotlib.pyplot as plt

num_pattern = re.compile(r'[-+]?\d*\.\d+[E][-+]?\d+')

data = {}  # put in this list
data['nw'] = 129
data['nh'] = 129

with open('iterhybrid_cocos02.eqdsk', 'r') as f:
    eq = f.readlines()
    p = []
    for line in eq[3465:]:
        found_numbers = num_pattern.findall(line)

        p.extend([float(x) for x in found_numbers])

plt.figure(figsize=(4, 6))
zs, rs = [], []
for i in range(0, len(p) - 1, 2):
    r, z = p[i], p[i + 1]
    if abs(z) > 3.99:
        print(i)
        plt.scatter(r, z, c='r', s=2)
    else:
        rs.append(r)
        zs.append(z)
        plt.scatter(r, z, c='g', s=2)

plt.grid()
plt.show()
print(data)

js = json.load(open('../PAM/template_D3D_1layer_2species.json'))

# update (Z, R)
js['equilibrium']['time_slice'][0]['boundary']['outline']['z'] = zs
js['equilibrium']['time_slice'][0]['boundary']['outline']['r'] = rs


json.dump(js, open('../PAM/ITER_cocos02.json', 'w'))
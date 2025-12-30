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
for i in range(0, len(p) - 1, 2):
    x, y = p[i], p[i + 1]
    plt.scatter(x, y, c='g', s=2)
plt.grid()
plt.show()
print(data)
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取
df = pd.read_pickle("simulation_results.pkl")
print(df['T_e'])
# 2. 访问标量
from test import visualize_real_data

for sta in ['s_face', 'T_e', 'n_e']:
    for idx in range(0, len(df[sta]), 100):
        visualize_real_data(df[sta][idx], sta + '/' + str(idx // 100))
        print(idx)
exit(0)
print(df['T_e'][:][0].shape)
plt.scatter(df['t'], [x[5] for x in df['s_face']])
plt.title("Q Fusion over time")
plt.show()

# 3. 访问向量
# df['T_e'] 这一列的每一个元素都是一个 numpy array
# 例如：取出最后时刻的温度剖面
final_Te_profile = df['T_e'].iloc[0]
plt.plot(final_Te_profile)
plt.title("Final Temperature Profile")
# plt.show()
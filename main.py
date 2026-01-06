from matplotlib import pyplot as plt
import numpy as np
from torax._src.config import config_loader
from run_loop_sim import prepare_simulation
from torax._src.config import build_runtime_params
from torax._src.orchestration.sim_state import SimState

config = config_loader.build_torax_config_from_file('config/ITER.py')

(
    initial_state,
    post_processed_outputs,
    step_fn,
) = prepare_simulation(config)

post_processing_history = [post_processed_outputs]
current_state = initial_state
override = build_runtime_params.RuntimeParamsProvider.from_config(
    config_loader.build_torax_config_from_file('config/PAM_pellet.py'))
cur_override = None

x = []
y = []
for t in range(15000):
    print('step', t)
    if 3000 < t < 4000 or 7000 < t < 8000:
        print('Inject')
        cur_override = override
    else:
        cur_override = None
    current_state, post_processed_outputs = step_fn(
        current_state,
        post_processing_history[-1],
        runtime_params_overrides=cur_override
    )
    post_processing_history.append(post_processed_outputs)
    print(current_state.core_profiles.T_e.value[0])
    x.append(t)
    y.append(current_state.core_profiles.T_e.value[0])

with open('data.txt', 'w') as f:
    for yi in y:
        f.write(str(yi) + '\n')

x = np.array(x)
y = np.array(y)
mask1 = (x > 3000) & (x < 4000)
mask2 = (x > 7000) & (x < 8000)
mask = mask1 | mask2

plt.xlabel('t')
plt.ylabel('T_e')
plt.scatter(x[~mask], y[~mask], s=1, c='b', label='Inject speed 1e22')
plt.scatter(x[mask], y[mask], s=1, c='r', label='No injection')
plt.grid(True)
plt.legend()
plt.show()

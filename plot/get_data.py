import numpy as np
from torax._src.config import config_loader
from run_loop_sim import prepare_simulation
from torax._src.config import build_runtime_params
from torax._src.orchestration.sim_state import SimState
import pandas as pd


config = config_loader.build_torax_config_from_file('config/ITER.py')

(
    initial_state,
    post_processed_outputs,
    step_fn,
) = prepare_simulation(config)

post_processing_history = [post_processed_outputs]
current_state = initial_state
override = build_runtime_params.RuntimeParamsProvider.from_config(
    config_loader.build_torax_config_from_file('config/PAM_pellet.py')
)
runtime_data = pd.DataFrame(
    columns = [
        't', 'Q_fusion', 'H98', 'W_thermal_total', 'q95', 'q_min',
        'q_face', 's_face', 'T_e', 'n_e'
    ]
)

'''
print(post_processed_outputs.Q_fusion)
print(post_processed_outputs.H98)
print(post_processed_outputs.W_thermal_total)
print(post_processed_outputs.q95)
print(post_processed_outputs.q_min)


print(current_state.core_profiles.q_face)
print(current_state.core_profiles.s_face)
print(current_state.core_profiles.T_e)
print(current_state.core_profiles.n_e)
# exit(0)
'''

cur_override = None
def get_vector(val):
    if val is None: return np.zeros(25) # 占位
    # 如果是 CellVariable (有 value 属性)，取 value
    if hasattr(val, 'value'):
        return np.array(val.value)
    # 否则直接转 numpy
    return np.array(val)

def get_scalar(val):
    if val is None: return 0.0
    # 如果是 JAX 数组，转为 float；如果是 python float，保持不变
    return val.item() if hasattr(val, 'item') else val

simulation_history = []
for t in range(10000):
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
    row_data = {
        't': t,

        'pellet': 1 if not cur_override else 0,
        'Q_fusion': get_scalar(post_processed_outputs.Q_fusion),
        'H98': get_scalar(post_processed_outputs.H98),
        'W_thermal_total': get_scalar(post_processed_outputs.W_thermal_total),
        'q95': get_scalar(post_processed_outputs.q95),
        'q_min': get_scalar(post_processed_outputs.q_min),


        'q_face': get_vector(current_state.core_profiles.q_face),
        's_face': get_vector(current_state.core_profiles.s_face),
        'T_e': get_vector(current_state.core_profiles.T_e),
        'n_e': get_vector(current_state.core_profiles.n_e)
    }
    simulation_history.append(row_data)
    post_processing_history.append(post_processed_outputs)

runtime_data = pd.DataFrame(simulation_history)
runtime_data.to_pickle("simulation_results.pkl")
print("Data saved to simulation_results.pkl (Recommended)")

csv_df = runtime_data.copy()
vector_cols = ['q_face', 's_face', 'T_e', 'n_e']
for col in vector_cols:
    csv_df[col] = csv_df[col].apply(lambda x: x.tolist())

csv_df.to_csv("simulation_results.csv", index=False)
print("Data saved to simulation_results.csv")

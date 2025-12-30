from torax._src.config import config_loader
from torax._src.orchestration import run_simulation

config = config_loader.build_torax_config_from_file('/Users/sheldonwang/Transport/iterhybrid_rampup.py')

for c in config:
    print(c)

data_tree, state_history = run_simulation.run_simulation(
      config,
      log_timestep_info=False,
      progress_bar=True,
)

print(data_tree)
print(state_history)
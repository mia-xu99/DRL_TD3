import os
import shutil
import yaml
from datetime import datetime

def create_experiment_dir(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg['experiment_id']
    exp_name = cfg['experiment_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = f"experiments/{exp_id}_{exp_name}_{timestamp}"
    os.makedirs(exp_dir)

    # Subdirs
    os.makedirs(os.path.join(exp_dir, "models"))
    os.makedirs(os.path.join(exp_dir, "tensorboard"))

    # Copy config
    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))

    # Init metrics.csv
    with open(os.path.join(exp_dir, "metrics.csv"), "w") as f:
        f.write("episode,steps,total_reward,success,collision,mean_dv,mean_dw,path_length\n")

    # Init notes.md
    with open(os.path.join(exp_dir, "notes.md"), "w") as f:
        f.write(f"# Experiment {exp_id} – {exp_name}\n\n")
        f.write("## Purpose\n\n## Observations\n\n## Results\n\n## Next Step\n")

    return exp_dir


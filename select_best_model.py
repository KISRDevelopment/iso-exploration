import json 
import sys 

hyperparam_opt_results = sys.argv[1]
output_path = sys.argv[2]

with open(hyperparam_opt_results, 'r') as f:
    cfgs = json.load(f)

best_cfg = min(cfgs, key=lambda c: c['mean_loss'])

with open(output_path, 'w') as f:
    json.dump(best_cfg, f, indent=4)
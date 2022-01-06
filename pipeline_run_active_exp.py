import opt_mp
import glob 
import json
import os  

exp_cfgs = glob.glob("exp_cfgs/*.json")
cfgs = [
    'cfgs/pi.json', 
    'cfgs/ei.json',
    'cfgs/greedy.json',
    'cfgs/eg.json',
    'cfgs/random.json']

for exp_cfg_path in exp_cfgs:
    exp_name = os.path.basename(exp_cfg_path).replace(".json","")
    for model_cfg_path in cfgs:

        cfg_name = os.path.basename(model_cfg_path).replace(".json", "")
        print("%s_%s" % (exp_name, cfg_name))

        with open(exp_cfg_path, 'r') as f:
            exp_cfg = json.load(f)

        exp_cfg['n_reps'] = 50

        with open(model_cfg_path, 'r') as f:
            model_cfg = json.load(f)

        opt_mp.main(exp_cfg, model_cfg, "traces/%s_%s.json" % (exp_name, cfg_name))
        
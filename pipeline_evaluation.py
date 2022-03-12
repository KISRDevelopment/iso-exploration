import opt_mp
import glob 
import json
import os  
import evaluate_model 

model_cfgs = glob.glob('cfgs/models/*.json')

os.makedirs('results_eval/', exist_ok=True)

for model_cfg_path in model_cfgs:

    basename = os.path.basename(model_cfg_path).replace('.json','')

    print("Running %s" % basename)
    evaluate_model.main(model_cfg_path, 'splits.npy', 'results_eval/%s.npz' % basename)

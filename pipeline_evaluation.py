import opt_mp
import glob 
import json
import os  
import evaluate_model 

dataset_path = 'r490.csv'
splits_path = 'r490_splits.npy'
output_path = 'tmp/r490_results'

model_cfgs = glob.glob('cfgs/models/*.json')

os.makedirs(output_path, exist_ok=True)

for model_cfg_path in model_cfgs:

    basename = os.path.basename(model_cfg_path).replace('.json','')

    print("Running %s" % basename)
    evaluate_model.main(dataset_path, model_cfg_path, splits_path, '%s/%s.npz' % output_path)

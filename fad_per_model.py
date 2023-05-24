
import torch
from diffusion import Diffusion
from tqdm import tqdm

from fad_score import FADWrapper
import os
import shutil
from utils import save_samples
import numpy as np
import json
import pandas as pd

# LABELS = {
#       "0": "DogBark",
#       "1": "Footstep",
#       "2": "GunShot",
#       "3": "Keyboard",
#       "4": "MovingMotorVehicle",
#       "5": "Rain",
#       "6": "Sneeze_Cough"
#     },


gt_embedding_path = os.path.join(os.getcwd(), 'fad_score', 'data', 'eval')
# create_labels = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough']
# create_labels = ['gunshot', 'moving_motor_vehicle', 'sneeze_cough']
# create_labels = ['moving_motor_vehicle']
create_labels = ['sneeze_cough']

model_number = '4' # '5' #    '3' # '6' #   '1'
training_kind =  'pretrained' # 'finetuned ' #'classwise' #  
schedule = 'quadratic' # 'quadratic_on_augmented' #  'linear_on_augmented_correct' # 'linear_on_augmented' # 'linear' # 'sigmoid' # 

models_address = os.path.join(os.getcwd(), 'models', f'model{model_number}', f'{training_kind}', f'{schedule}', 'pretrain_on_01356')
# config_json = json.load(open(os.path.join(os.getcwd(),  'models', f'model{model_number}', f'{training_kind}', f'{schedule}', 'finetune4.json')))
config_json = json.load(open(os.path.join(models_address, 'pretrain.json')))

# models_address = '/share/home/patricia/Projects/dcase23_foley/AudioDiffusion/models/model5/finetuned/quadratic_on_augmented/tmp_eval'
# config_json = json.load(open('/share/home/patricia/Projects/dcase23_foley/AudioDiffusion/models/model5/finetuned/quadratic_on_augmented/tmp_eval/finetune6.json'))

def fad(model, config, diffusion, labels = [0,1,2,3,4,5,6]): # labels are the classes you want to generate for. Currently you also have to set "name_of_sound_list" manually
    
    result_name = f'model{model_number}_{training_kind}_{schedule}_{create_labels[0]}'
    config.output_path = '_generated_samples_eval/{}/generated_files'.format(result_name)
    config.create_count = 10
    config.create_loop = 1
    if os.path.exists(config.output_path):
        shutil.rmtree(config.output_path)
    os.makedirs(config.output_path)
    for label in labels:
        config.create_label = label
        for i in range(10): # the reason behind this for loop is that my RAM didn't have enough space to save 100 samples
            torch.cuda.empty_cache()
            save_samples(model, diffusion, config, it=i)
            print(f'-- generated {(i+1)*10} samples')
    print('calling fad wrapper')
    fad_wrapper = FADWrapper.FADWrapper(generated_audio_samples_dir=config.output_path, ground_truth_audio_samples_dir=gt_embedding_path, name_of_sound_list=create_labels, result_dir=result_name) #change the name list based on your classes
    # fad_wrapper = FADWrapper.FADWrapper(generated_audio_samples_dir=config.output_path, ground_truth_audio_samples_dir="fad_score/data/eval", name_of_sound_list=["footstep", "rain"]) #change the name list based on your classes
    fd = fad_wrapper.compute_fad()
    print(f'computed fad {fd}')
    return list(fd['FAD'])




dir = os.listdir(models_address)
# create the config file
class Config:
    pass
config = Config()
for key, data in config_json.items():
    setattr(config, key, data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.device = device

if config.model_number == 1:
    from model import UNet
elif config.model_number == 2:
    from model2 import UNet
elif config.model_number == 3:
    from model3 import UNet
elif config.model_number == 4:
    from model4 import UNet
elif config.model_number == 5:
    from model5 import UNet
elif config.model_number == 6:
    from model6 import UNet
    
model = UNet(device, config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
diffusion = Diffusion(config)

results= {}
results['name'] = []
results['fad'] = []
results['model'] = []
for i, m in tqdm(enumerate(dir)):
    print('*' * 42)
    print(f'Evaluating model {i+1}: {m}')
    # print(f'Evaluating model {i+1}: {models_address} -- {m}')
    # print('{}/{}'.format(models_address, m))
    try: #not all models were trained based on model 4. If we fail to load the model we assume it had been a different model, and move it to a seaprate folder.
        loaded = torch.load('{}/{}'.format(models_address, m))
        model.load_state_dict(loaded['model'])
        print('model loaded!')
        if not config.change_lr:
            optimizer.load_state_dict(loaded['optimizer'])
        config.current_epoch = loaded['epoch']
        print(f'generating samples and computing fad scores for class {create_labels}...')
        fad_scores = fad(model, config, diffusion, labels=[6]) # 4 moving motor vehicle, 6 sneeze cough
        print(f'--calculated score: {fad_scores}')
        results['name'].append(m)
        results['model'].append(config.model_number)
        results['fad'].append(fad_scores)
        print(f'{m} scores {fad_scores} for class {create_labels}')
        pd.DataFrame.from_dict(results).to_csv(f'results_model{model_number}_{training_kind}_01356_{schedule}_gen6.csv')
        print('appended to df')
        # os.rename("all_trained_models/classwise/quadratic/{}".format(m), "done/{}".format(m))
    except:
        print('failed to load!')
        # os.rename("all_trained_models/classwise/quadratic/{}".format(m), "not_4/{}".format(m))



# finetune-linear-5on4augm-corr_pretrain950+_0eps.p scores [47.99680545002994] for class ['moving_motor_vehicle']
# finetune-linear-5on4augm-corr_pretrain950+_50eps.p scores [49.40865963094792] for class ['moving_motor_vehicle']
# finetune-linear-5on4augm-corr_pretrain950+_100eps.p scores [40.25413306917275] for class ['moving_motor_vehicle']
# finetune-linear-5on4augm-corr_pretrain950+_150eps.p scores [49.750653680710315] for class ['moving_motor_vehicle']
# 



# Train with variance=.2 scheduler again for class6
# Ask Tara for performance of model4
# 
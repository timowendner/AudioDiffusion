
import torch
# from model4 import UNet  # import your own module(s) here\
from model5 import UNet
from diffusion import Diffusion
from tqdm import tqdm

from fad_score import FADWrapper
import os
import shutil
from utils import save_samples
import numpy as np
import json
import pandas as pd

generator = torch.Generator().manual_seed(42)

gt_embedding_path = os.path.join(os.getcwd(), 'fad_score', 'data', 'eval')

def fad(model, config, diffusion, labels = [0,1,2,3,4,5,6]): # labels are the classes you want to generate for. Currently you also have to set "name_of_sound_list" manually
    config.output_path = '{}/generated_files'.format(config.model_name)
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
    fad_wrapper = FADWrapper.FADWrapper(generated_audio_samples_dir=config.output_path, ground_truth_audio_samples_dir=gt_embedding_path, name_of_sound_list=["moving_motor_vehicle", "sneeze_cough"]) #change the name list based on your classes
    # fad_wrapper = FADWrapper.FADWrapper(generated_audio_samples_dir=config.output_path, ground_truth_audio_samples_dir="fad_score/data/eval", name_of_sound_list=["footstep", "rain"]) #change the name list based on your classes
    fd = fad_wrapper.compute_fad()
    print(fd)
    return list(fd['FAD'])



# models_address = 'all_trained_models'
# dir = os.listdir(models_address)
# config_json = json.load(open('config4.json'))


# to be running on fs and rk
# models_address = os.path.join(os.getcwd(), 'all_trained_models', 'classwise', 'quadratic') 
# config_json = json.load(open(os.path.join(os.getcwd(), 'configs', 'config_t4.json')))

models_address = os.path.join(os.getcwd(), 'all_trained_models', 'classwise', 'linear')
config_json = json.load(open(os.path.join(os.getcwd(), 'configs', 'config_t4_linear.json')))

# models_address = os.path.join(os.getcwd(), 'all_trained_models', 'pretrained')
# config_json = json.load(open(os.path.join(os.getcwd(), 'configs', 'config_pretrain.json')))

dir = os.listdir(models_address)
# create the config file
class Config:
    pass
config4 = Config()
for key, data in config_json.items():
    setattr(config4, key, data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config4.device = device

model4 = UNet(device, config4).to(device)
optimizer4 = torch.optim.Adam(model4.parameters(), lr=config4.lr)
diffusion = Diffusion(config4)
results= {}
results['name'] = []
results['fad'] = []
results['model'] = []
for i, m in tqdm(enumerate(dir)):
    print('*' * 42)
    print(f'Evaluating model {i+1}: {models_address} -- {m}')
    # print('{}/{}'.format(models_address, m))
    try: #not all models were trained based on model 4. If we fail to load the model we assume it had been a different model, and move it to a seaprate folder.
        loaded = torch.load('{}/{}'.format(models_address, m))
        model4.load_state_dict(loaded['model'])
        print('model loaded!')
        if not config4.change_lr:
            optimizer4.load_state_dict(loaded['optimizer'])
        config4.current_epoch = loaded['epoch']
        print('generating...')
        fad_scores = fad(model4, config4, diffusion, labels=[4,6]) # 4 moving motor vehicle, 6 sneeze cough
        results['name'].append(m)
        results['model'].append('4')
        results['fad'].append(fad_scores)
        # pd.DataFrame.from_dict(results).to_csv('results_quadratic_classwise.csv')
        # pd.DataFrame.from_dict(results).to_csv('results_linear_classwise.csv')
        pd.DataFrame.from_dict(results).to_csv('results_linear_denoised=5_classwise.csv')
        # os.rename("all_trained_models/classwise/quadratic/{}".format(m), "done/{}".format(m))
    except:
        print('failed to load!')
        os.rename("all_trained_models/classwise/quadratic/{}".format(m), "not_4/{}".format(m))



    # "label_path": {
    #   "0": "DogBark",
    #   "1": "Footstep",
    #   "2": "GunShot",
    #   "3": "Keyboard",
    #   "4": "MovingMotorVehicle",
    #   "5": "Rain",
    #   "6": "Sneeze_Cough"
    # },
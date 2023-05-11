
import torch
from model4 import UNet  # import your own module(s) here
from diffusion import Diffusion

from fad_score import FADWrapper
import os
import shutil
from utils import save_samples
import json
import random

config_json = json.load(open('config.json'))
class_name = "moving_motor_vehicle"
class Config:
    pass
config = Config()
for key, data in config_json.items():
    setattr(config, key, data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.device = device

model_address = ['09May_2319.p', '09May_2331.p', '11May_1028.p', '09May_2348.p','09May_2225.p','09May_2243.p','09May_0316.p','09May_0523.p']#, '09May_0158.p', '07May_1146.p','09May_2056.p','09May_2201.p']
#model_address = ['09May_2319.p', '09May_2331.p', '09May_2342.p', '09May_2348.p','09May_2225.p','09May_2243.p','09May_0316.p','09May_0523.p', '09May_0158.p', '07May_1146.p','09May_2056.p','09May_2201.p','09May_2056.p','09May_2201.p']

def FAD(saved_address):
    fad_wrapper = FADWrapper.FADWrapper(generated_audio_samples_dir=saved_address, ground_truth_audio_samples_dir="fad_score/data/eval", name_of_sound_list=[class_name])
    fd = fad_wrapper.compute_fad()
    print(fd)
    return fd['FAD']


def generate(models, diffusion,label, gen_num=1000):
    config.create_count = 10
    #if os.path.exists(config.output_path):
    #    shutil.rmtree(config.output_path)
    #os.makedirs(config.output_path)
    config.create_label = label
    for i in range(int(gen_num/config.create_count)):
        torch.cuda.empty_cache()
        config.create_loop = random.randint(2, 5)
        model = models[random.randint(0, len(models)-1)]
        save_samples(model, diffusion, config, it=i)


def select(pick_num=100, trials = 200):
    dir = os.listdir(config.output_path+'/'+class_name)
    trial_path = config.output_path + '/trials/'
    seleced_path = config.output_path + '/selected/'
    os.makedirs(seleced_path + class_name, exist_ok=True)
    fads = []

    best_fad = 100
    best_set = []
    for _ in range(trials):
        print(_)
        os.makedirs(trial_path + class_name, exist_ok=False)
        res = random.sample(range(1, len(dir)), pick_num)
        for i,f in enumerate(dir):
            if i not in res:
                continue
            else:
                shutil.copyfile(config.output_path + '/' + class_name + '/' + f, trial_path + class_name + '/' + f)
        
        score = FAD(trial_path)
        fads.append(score)
        if score[0] < best_fad:
            best_fad = score
            best_set = res
        shutil.rmtree(trial_path)
    for i,f in enumerate(dir):
        if i not in best_set:
            continue
        else:
            shutil.copyfile(config.output_path + '/' + class_name + '/' + f, seleced_path + class_name + '/' + f)
    print(best_fad)
    print(fads)



def main():
    models = []
    optims = []
    for m in model_address:
        temp = UNet(device, config).to(device)
        temp_optimizer = torch.optim.Adam(temp.parameters(), lr=config.lr)
        temp_loaded = torch.load(m)
        temp.load_state_dict(temp_loaded['model'])
        if not config.change_lr:
            temp_optimizer.load_state_dict(temp_loaded['optimizer'])
        models.append(temp)
        optims.append(temp_optimizer)
    #model = UNet(device, config).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    diffusion = Diffusion(config)
    #loaded = torch.load(model_address)
    #model.load_state_dict(loaded['model'])
    print('models loaded!')
    #generate(models, diffusion ,4)
    select()





main()
To use the model at first clone the repository
```bash
!git clone https://github.com/timowendner/AudioDiffusion.git
```

To get to the right branch use:
```bash
!cd folderpath/AudioDiffusion && git pull && git checkout timo2
```

At first create the Config file below and then run:
```bash
!cd AudioDiffusion && python main.py --config_path='/content/config.json'
```
the args can define:
 - --config_path='path': Specify the config path
 - --*train*: Specify if the model should be trained 
 - --*load*: Specify if the last modified model should be loaded 
 - --*lr=0.0001*: Change the learning rate

Before using a model we have to define a Json config file with the following specifications:
 - model_name: the name of the model
 - data_path: the path of the data
 - model_path: the path of the model
 - output_path: the path of the output
 - label_path: the path of the labels
 - label_count: total count of labels
 - label_train: the labels that we want to train on (i.e. [0] would just train on label 0)
 - step_count: the diffusion steps that we add
 - beta_start: the starting point of the diffusion
 - beta_end: the endpoint of the diffusion
 - beta_schedule: what type of schedule to apply can be 'quadratic', 'linear'
 - create_label: what label to create while generating a sample
 - create_count: how many samples to create
 - create_loop: how often every timestamp gets applied while generating a sample
 - create_last: specify how often the last timestamp will be applied
 - audio_length: The length of the audio-files
 - model_layers: Specify the layers of the UNet
 - model_out: Specify the last convolution of the UNet
 - model_kernel: Specify the kernel used for the convolution
 - model_scale: Specify how much the data is downscaled in every layer
 - save_time: after how many seconds the model will be saved
 - lr: learning rate
 - num_epochs: the number of epochs that the model should be trained

As an example we have here:
```python
import json

# Create a dictionary to be saved as JSON
data = {
    "model_name": "line",
    "data_path": "/content/drive/MyDrive/AudioDiffusion/data",
    "model_path": "/content/drive/MyDrive/AudioDiffusion/models",
    "output_path": "/content/drive/MyDrive/AudioDiffusion/output",
    "label_path": {
      0: "DogBark",
      1: "Footstep",
      2: "GunShot",
      3: "Keyboard",
      4: "MovingMotorVehicle",
      5: "Rain",
      6: "Sneeze_Cough"
    },
    "label_train": [2],
    "step_count": 250,
    "beta_start": 0.00001,
    "beta_end": 0.02,
    "beta_schedule": "quadratic",
    "create_label": 2,
    "create_count": 100,
    "create_loop": 2,
    "create_last": 0,
    "audio_length": 88200,
    "model_layers": [64, 64, 96, 128],
    "model_out": [64, 64, 32],
    "model_kernel": 9,
    "model_scale": 4,
    "save_time": 200,
    "lr": 0.001,
    "num_epochs": 1000
}

# Save dictionary as JSON to a file
with open("config.json", "w") as f:
    json.dump(data, f)
```
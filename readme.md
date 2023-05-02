In order to calculate FAD scores for a directory of models, change the following parameters in fad_per_model.py:

1. import your own model
2. set "models_address" to your model directory.
3. change config address
4. Currently, due to some address issues you need a "data" directory in the main directory that includes the data (audio files) as well as the pretrained VGG model, and a data directory in "fad_score" with the rest of the eval folder. (ToDo: fix this.)
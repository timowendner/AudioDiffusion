import optuna
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam
from model_ph import UNet  # import your own module(s) here
from dataloader import AudioDataset
from diffusion import Diffusion


generator = torch.Generator().manual_seed(42)

def objective(trial):
    # Define the hyperparameters to search over
    config = {
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'step_count': trial.suggest_int('step_count', 1, 10),
        'label_count': trial.suggest_int('label_count', 1, 10),
    }

    # Define your data loading and splitting code here
    dataset = AudioDataset(Diffusion.diffusion, config, model.device)
    train_data, val_data = random_split(dataset, [0.7,0.3], generator=generator)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])

    # Define the model and the optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(device, config)
    model.to(device)
    optimizer = config['optimizer'](model.parameters(), lr=config['lr'])

    # Train the model and calculate the validation loss
    criterion = nn.MSELoss()
    
    for epoch in range(100):
        model.train()
        for x, t, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x.to(device), t.to(device), y.to(device))
            loss = criterion(y_pred, y.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(x.to(device), t.to(device), y.to(device)), y.to(device)) for x, t, y in val_loader) / len(val_loader)

    return val_loss.item()


if __name__ == '__main__':
    # Run the hyperparameter search using Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters and the corresponding validation loss
    print(f'Best hyperparameters: {study.best_params}')
    print(f'Best validation loss: {study.best_value}')


# Use this sweep configuration to initialize a sweep in wandb
import wandb
from cmcd import train_sweep

sweep_config = {
    'method': 'random',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'T': {
            'value': 50
        },
        'step_size': {
            'values': [0.01, 0.05, 0.1]
        },
        'n_steps': {
            'value': 50
        },
        'n_batch': {
            'values': [64, 128, 256]
        },
        'lr': {
            'values': [0.001, 0.0003, 0.0001]
        },
        'hidden_dim': {
            'values': [32, 64, 128]
        },
        'n_layers': {
            'values': [2, 3, 4]
        },
        'dataset': {
            'value': "gmm"
        },
        'checkpoint_dir': {
            'value': "./"
        },
        'seed': {
            'value': 0
        },
        'use_kl_loss': {
            'values': [False, True]
        },
        'num_samples': {
            'value': 10000
        },
        'n_steps_eval': {
            'value': 10
        },
        'n_samples_eval': {
            'value': 4096
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="master-project")


# Start the sweep agent
wandb.agent(sweep_id, function=train_sweep)

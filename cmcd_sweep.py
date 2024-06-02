
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
            'value': 0.05
        },
        'n_steps': {
            'values': [50, 100, 500]
        },
        'n_batch': {
            'values': [64, 128, 512]
        },
        'lr': {
            'values': [0.001, 0.0003]
        },
        'hidden_dim': {
            'value': 128
        },
        'n_layers': {
            'values': [3, 4]
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
        'n_samples_train': {
            'values': [128, 1024, 10_000]
        },
        'n_steps_eval': {
            'value': 5
        },
        'n_samples_val': {
            'value': 4096
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="master-project")


# Start the sweep agent
wandb.agent(sweep_id, function=train_sweep)

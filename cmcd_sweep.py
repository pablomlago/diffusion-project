
# Use this sweep configuration to initialize a sweep in wandb
import wandb
from cmcd import train_sweep
from absl import app, flags

# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_enum("sweep_dataset", "gmm", ["gmm", "normal", "hard_gmm"], "Dataset choice")

sweep_config_gmm = {
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
            'values': [0.05, 0.01]
        },
        'n_steps': {
            'value': 800
        },
        'n_batch': {
            'value': 128
        },
        'optimiser': {
            'values': ['adam', 'adamw']
        },
        'lr': {
            'value': 0.0003
        },
        'hidden_dim': {
            'value': 128
        },
        'n_layers': {
            'value': 3
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
        'use_double_drift': {
            'values': [False, True]
        },
        'n_samples_train': {
            'value': 64
        },
        'n_steps_eval': {
            'value': 5
        },
        'n_samples_val': {
            'value': 4096
        },
        'n_samples_test': {
            'value': 10_000
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "eta": 3,
        "min_iter": 100,
    },
}

sweep_config_hard_gmm = {
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
            'values': [0.001, 0.005]
        },
        'n_steps': {
            'value': 800
        },
        'n_batch': {
            'value': 128
        },
        'optimiser': {
            'values': ['adam', 'adamw']
        },
        'lr': {
            'values': [0.0003, 0.001, 0.0001]
        },
        'hidden_dim': {
            'value': 128
        },
        'n_layers': {
            'values': [3, 4]
        },
        'dataset': {
            'value': "hard_gmm"
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
        'use_double_drift': {
            'values': [False, True]
        },
        'n_samples_train': {
            'value': 128
        },
        'n_steps_eval': {
            'value': 5
        },
        'n_samples_val': {
            'value': 4096
        },
        'n_samples_test': {
            'value': 10_000
        }
    },
    "early_terminate": {
        "type": "hyperband",
        "eta": 3,
        "min_iter": 100,
    },
}


def main(argv):
    if FLAGS.sweep_dataset in ["gmm", "normal"]:
        sweep_config = sweep_config_gmm
    elif FLAGS.sweep_dataset in ["hard_gmm"]:
        sweep_config = sweep_config_hard_gmm
    sweep_id = wandb.sweep(sweep_config, project=f"master-project-{FLAGS.sweep_dataset}")
    # Start the sweep agent
    wandb.agent(sweep_id, function=train_sweep)

if __name__ == '__main__':
    app.run(main)
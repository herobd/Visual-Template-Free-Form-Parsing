# Pairing project
based on victoresque pytorch template

## Requirements
* Python 3.x
* PyTorch


## Folder Structure
  ```
  
  │
  ├── train.py - example main
  ├── config_ex.json - example config file
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── datasets/ - default datasets folder
  │   └── ai2d.py - loads AI2D dataset for query-response masking
  │
  ├── logger/ - for training process logging
  │   └── logger.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── modules/ - submodules of your model
  │   ├── loss.py
  │   ├── metric.py
  │   ├── model.py
  │   └── unet.py - UNet
  │
  ├── saved/ - default checkpoints folder
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      └── ...
  ```

### Config file format
Config files are in `.json` format:
  ```
  {
    "name": "Mnist_LeNet",        // training session name
    "cuda": true,                 // use cuda
    "override": false,            // if resuming, whether to replace the previous config with this one
    "data_loader": {
        "data_set_name":          // the name of the dataset
        "data_dir": "datasets/",  // dataset path
        "batch_size": 32,         // batch size
        "shuffle": true,          // shuffle data each time calling __iter__()
        "num_workers": 2          // number of workers loading data (0 means on main thread, which is probably bad)
    },
    "validation": {
        "validation_split": 0.1,  // validation data ratio
        "shuffle": true           // shuffle training data before splitting
    },
    "optimizer_type": "Adam",
    "optimizer": {
        "lr": 0.001,              // (optional) learning rate
        "weight_decay": 0         // (optional) weight decay
    },
    "loss": "my_loss",            // loss
    "metrics": [                  // metrics, function names
      "my_metric",
      "my_metric2"
    ],
    "trainer": {
        "epochs": 1000,           // number of training epochs
        "save_dir": "saved/",     // checkpoints are saved in save_dir/name
        "save_step": 5000,        // save checkpoints every save_step iterations
        "val_step": 5000,         // run on validation set every val_step iterations
        "log_step": 1000,         // print log statement every log_step iterations
        "verbosity": 2,           // 0: quiet, 1: per epoch, 2: full
        "monitor": "val_loss",    // monitor value for best model
        "monitor_mode": "min"     // "min" if monitor value the lower the better, otherwise "max", "none" if you dont want to save best
    },
    "arch": "MnistModel",         // model architecture
    "model": {}                   // model configs
  }
  ```

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```


The checkpoints will be saved in `save_dir/name`.

The config file is saved in the same folder. (as a reference only, the config is loaded from the checkpoint)

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
  }
  ```


# Pairing project
based on victoresque pytorch template

## Requirements
* Python 3.x (at least 3.7 for doing optimization)
* PyTorch 1.0+

I used conda, but it shouldn't be hard to install the packages another way.

# Installing to get it to work with pytorch 1
First check gcc version (must be atleast 4.9) and cuda version (8 requires gcc to be 5.3 or lower).
If upgrading cuda, remove the old version first either with apt-get or uninstall script in /usr/cuda/bin.
    (Be sure CUDA_HOME and PATH are right after installation)

`conda install $`
* `ipython`
* `pytorch torchvision -c pytorch` or what ever the command is on pytorch.org given cuda version, etc
* `opencv`
* `scikit-image`


for cvxpy (optimization), you must have python 3.7 or later:

clone github: https://github.com/cvxgrp/cvxpy

`python setup.py install` in the cvxpy repo

# Install
`python setup.py build develop`

## Reproducability instructions


### Setting up dataset 
`../data/forms/`

### Pretraining detector network
`python train.py -c cf_detector.json`

### Training pairing network
`python train.py -c cf_pairing.json`

### Evaluating

#### Standard experiments

If you want to run on GPU, add `-g #`, where `#` is the GPU number.

Remove the `-T` flag to run on the validation set.


Detection, full set: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth.tar -n 0 -T`

Detection, pairing set: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth.tar -n 0 -T -a data_loader=special_dataset=simple`

Pairing, no optimization: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth.tar -n 0 -T`
`

Pairing, with optimization: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth.tar -n 0 -T -a optimize=true`

#### Perfect information experiments

Pairing, GT detections: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth.tar -n 0 -T -a useDetect=gt`

Pairing, optimized with GT num neighnors:  `python eval.py -c saved/pairing/checkpoint-iteration125000.pth.tar -n 0 -T -a optimize=gt`

### Training baseline models

#### Detector using regular convs
`python train.py -c cf_baseline_detector.json`

#### Classifier using non-visual features

Make training data for no visual feature pairing: 
1. `mkdir out`
2. `python eval.py -c saved/saved/pairing/checkpoint-iteration125000.pth.tar -g 0 -n 10000 -a save_json=out/detection_data,data_loader=batch_size=1,data_loader=num_workers=0,data_loader=rescale_range=0.52,data_loader=crop_params=,validation=rescale_range=0.52,validation=crop_params=`

Train no visual feature pairing: `python train.py -c cf_no_vis_pairing.json`

### Evaluating baseline models

Detection with regular convs, full set: `python eval.py -c saved/baseline_detector/checkpoint-iteration150000.pth.tar -n 0 -T`

Detection with regular convs, pairing set: `python eval.py -c saved/baseline_detector/checkpoint-iteration150000.pth.tar -n 0 -T -a data_loader=special_dataset=simple`


Distance based pairing: `python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a rule=closest`

Scoring functions pairing: `python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a rule=icdar`

No visual features pairing: `python eval.py -f cf_test_no_vis_pairing.json -n 0 -T`

No visual features pairing, with optimization: `python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a optimize=true`

#### Perfect information experiments

For GT detections:

`python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a rule=closest,useDetect=gt`

`python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a rule=icdar,useDetect=gt`

`python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a useDetect=gt`

For optimization with GT num neighbors:

`python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a rule=closest,optimize=gt`

`python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a rule=icdar,optimize=gt`

`python eval.py -f cf_test_no_vis_pairing.json -n 0 -T -a optimize=gt`




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
  ├── data_loader/ - 
  │   └── data_loaders.py - This provides access to all the dataset objects
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
    "gpu": 0,                       //gpu to use (only single device is supportted right now)
    "override": true,            // if resuming, whether to replace the previous config with this one
    "save_mode": "state_dict",  //whether to save just the state dict (recommended) or the whole model object (doesn't always work)
    "super_computer":false,     //whether to print inplace iteration number (doesn't work with slurm logging)
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


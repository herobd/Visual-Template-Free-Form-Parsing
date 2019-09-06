# Pairing project

This code is free to use for non-commericial purposes. Contact me if commericialization is desired.

## Requirements
* Python 3.x (at least 3.7 for doing optimization)
* PyTorch 1.0+

I used conda, but it shouldn't be hard to install the packages another way.

## Installing
First check gcc version (must be atleast 4.9) and cuda version (8 requires gcc to be 5.3 or lower).
If upgrading cuda, remove the old version first either with apt-get or uninstall script in /usr/cuda/bin.
    (Be sure CUDA_HOME and PATH are right after installation)

`conda install $`
* `ipython`
* `pytorch torchvision -c pytorch` or what ever the command is on pytorch.org given cuda version, etc
* `opencv`
* `scikit-image`


Run this in the Visual-Template-Free-Form-Parsing directoty: `python setup.py build develop`

###CVXPY

for cvxpy (optimization), you must have python 3.7 or later:

clone github: https://github.com/cvxgrp/cvxpy

`python setup.py install` in the cvxpy repo`

## Reproducability instructions


### Setting up dataset 
see https://github.com/herobd/NAF_dataset

### Pretraining detector network
`python train.py -c cf_detector.json`

### Training pairing network
`python train.py -c cf_pairing.json`

### Evaluating

#### Standard experiments

If you want to run on GPU, add `-g #`, where `#` is the GPU number.

Remove the `-T` flag to run on the validation set.


Detection, full set: `python eval.py -c saved/detector/checkpoint-iteration150000.pth -n 0 -T`

Detection, pairing set: `python eval.py -c saved/detector/checkpoint-iteration150000.pth -n 0 -T -a data_loader=special_dataset=simple`

Pairing, no optimization: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth -n 0 -T`
`

Pairing, with optimization: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth -n 0 -T -a optimize=true`

#### Perfect information experiments

Pairing, GT detections: `python eval.py -c saved/pairing/checkpoint-iteration125000.pth -n 0 -T -a useDetect=gt`

Pairing, optimized with GT num neighnors:  `python eval.py -c saved/pairing/checkpoint-iteration125000.pth -n 0 -T -a optimize=gt`

### Training baseline models

#### Detector using regular convs

`python train.py -c cf_baseline_detector.json`

Note: This will take a while before it begins training on your first run as it caches smaller sizes of the dataset.

#### Classifier using non-visual features

Make training data for no visual feature pairing: 
1. `mkdir out`
2. `python eval.py -c saved/detector/checkpoint-iteration150000.pth -g 0 -n 10000 -a save_json=out/detection_data,data_loader=batch_size=1,data_loader=num_workers=0,data_loader=rescale_range=0.52,data_loader=crop_params=,validation=rescale_range=0.52,validation=crop_params=,data_loader=cache_resized_images=0`

Train no visual feature pairing: `python train.py -c cf_no_vis_pairing.json`

### Evaluating baseline models

Detection with regular convs, full set: `python eval.py -c saved/baseline_detector/checkpoint-iteration150000.pth -n 0 -T`

Detection with regular convs, pairing set: `python eval.py -c saved/baseline_detector/checkpoint-iteration150000.pth -n 0 -T -a data_loader=special_dataset=simple`


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


## Usage

### train.py

This is the script that executes training based on a configuration file. The training code is found in `trainer/`. The config file specifies which trainer is used.

The usage is: `python train.py -c CONFIG.json`  (see below for example config file)

A training session can be resumed with: `python train.py -r CHECKPOINT.pth`

If you want to override the config file on a resume, just use the `-c` flag and be sure it has `"override": true`


### eval.py

This script runs a trained model (from a snapshot) through the dataset and prints its scores. It is also used to save images with the predictions on them.

Usage:  `python eval.py -c CHECKPOINT.pth -f OVERRIDE_CONFIG.pth -g (gpu number) -n (number of images to save) -d (directory to save images) -T`

The only flags required is `-c` or `-f`.

If `-T` is ommited it will run on the validation set instead of the test set.

There is an additional `-a` flag which allows overwriting of specific values of the config file using this format `key1=nestedkey=value,key2=value`. It also allows setting these special options (which are part of config):

Evaluating detector:
* `-a pretty=true`: Makes printed pictured cleaner (less details)
* `-a save_json=path/to/dir`: Save the detection results as jsons matching the dataset format.
* `-a THRESH=[float]`: Modify the threshold for displaying and precision/recall calculations. Default is 0.92

Evaluatring pairing:
* `-a useDetect=[gt,path]`:  Whether to use GT detections (`gt`) or can be directory with jsons with saved detections.
* `-a rule=[closest,icdar]`: Use a rule (nearest or scoring functions) to do pairing (instead of model).
* `-a optimize=[true,gt]`: Use optimization. If `gt` specified it will use the GT number of neighbors.
* `-a penalty=[float]`: The variable *c* in Equation 1. Default is 0.25
* `-a THRESH=[float]`: Modify the thresh for calculating prec/recall for relationships. Also is *T* in Equation 1. Default is 0.7
* `-a sweep_threshold=true`: Run metrics using a range of thresholds
* `-a draw_thresh=[float]`: Seperate threshold for which relationships get saved in images.
* `-a confThresh=[float]`: Threshold used for detections.
* `-a pretty=[true,light,clean]`: Different ways of displaying the results.

## File Structure

This is based on victoresque's pytorch template: https://github.com/victoresque/pytorch-template

  ```
  
  │
  ├── train.py - Training script
  ├── eval.py - Evaluation and display script
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
  │   ├── box_detect.py - base class for detection
  │   ├── forms_box_detect.py - detection for NAF dataset
  │   ├── forms_feature_pair.py - dataset for training non-visual classifier
  │   ├── graph_pair.py - base class for pairing
  │   ├── forms_graph_pair.py - pairing for NAD dataset
  │   └── test*.py - scripts to test the datasets and display the images for visual inspection
  │
  ├── logger/ - for training process logging
  │   └── logger.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── binary_pair_real.py - Provides classifying network for pairing and final prediction network for detection. Also can have secondary using non-visual features only classifier
  │   ├── coordconv.py - Implements a few variations of CoordConv. I didn't get better results using it.
  │   ├── csrc/ - Contains Facebook's implementation for ROIAlign from https://github.com/facebookresearch/maskrcnn-benchmark
  │   ├── roi_align.py - End point for ROIAlign code
  │   ├── loss.py - Imports all loss functions
  │   ├── net_builder.py - Defines basic layers and interpets config syntax into networks.
  │   ├── optimize.py - pairing descision optimization code
  │   ├── pairing_graph.py - pairing network class
  │   ├── simpleNN.py - defines non-convolutional network
  │   ├── yolo_box_detector.py - detector network class
  │   └── yolo_loss.py - loss used by detector
  │
  ├── saved/ - default checkpoints folder
  │
  ├── trainer/ - trainers
  │   ├── box_detect_trainer.py - detector training code
  │   ├── feature_pair_trainer.py - non-visual pairing training code
  │   ├── graph_pair_trainer.py - pairing training code
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      └── ...
  ```

### Config file format
Config files are in `.json` format. Example:
  ```
{
    "name": "pairing",                      # Checkpoints will be saved in saved/name/checkpoint-...pth
    "cuda": true,                           # Whether to use GPU
    "gpu": 0,                               # GPU number. Only single GPU supported.
    "save_mode": "state_dict",              # Whether to save/load just state_dict, or whole object in checkpoint
    "override": true,                       # Override a checkpoints config
    "super_computer":false,                 # Whether to mute training info printed
    "data_loader": {
        "data_set_name": "FormsGraphPair",  # Class of dataset
        "special_dataset": "simple",        # Use partial dataset. "simple" is the set used for pairing in the paper
        "data_dir": "../data/NAF_dataset",  # Directory of dataset
        "batch_size": 1,
        "shuffle": true,
        "num_workers": 1,
        "crop_to_page":false,
        "color":false,
        "rescale_range": [0.4,0.65],        # Form images are randomly resized in this range
        "crop_params": {
            "crop_size":[652,1608],         # Crop size for training instance
	    "pad":0
        },
        "no_blanks": true,                  # Removed fields that are blank
        "swap_circle":true,                 # Treat text that should be circled/crossed-out as pre-printed text
        "no_graphics":true,                 # Images not considered elements
        "cache_resized_images": true,       # Cache images at maximum size of rescale_range to make reading them faster
        "rotation": false,                  # Bounding boxes are converted to axis-aligned rectangles
        "only_opposite_pairs": true         # Only label-value pairs


    },
    "validation": {                         # Enherits all values from data_loader, specified values are changed
        "shuffle": false,
        "rescale_range": [0.52,0.52],
        "crop_params": null,
        "batch_size": 1
    },

    
    "lr_scheduler_type": "none",
 
    "optimizer_type": "Adam",
    "optimizer": {                          # Any parameters of the optimizer object go here
        "lr": 0.001,
        "weight_decay": 0
    },
    "loss": {                               # Name of functions (in loss.py) for various components
        "box": "YoloLoss",                  # Detection loss
        "edge": "sigmoid_BCE_loss",         # Pairing loss
        "nn": "MSE",                        # Num neighbor loss
        "class": "sigmoid_BCE_loss"         # Class of detections loss
    },
    "loss_weights": {                       # Respective weighting of losses (multiplier)
        "box": 1.0,
        "edge": 0.5,
        "nn": 0.25,
        "class": 0.25
    },
    "loss_params": 
        {
            "box": {"ignore_thresh": 0.5,
                    "bad_conf_weight": 20.0,
                    "multiclass":true}
        },
    "metrics": [],
    "trainer": {
        "class": "GraphPairTrainer",        # Training class name 
        "iterations": 125000,               # Stop iteration
        "save_dir": "saved/",               # save directory
        "val_step": 5000,                   # Run validation set every X iterations
        "save_step": 25000,                 # Save distinct checkpoint every X iterations
        "save_step_minor": 250,             # Save 'latest' checkpoint (overwrites) every X iterations
        "log_step": 250,                    # Print training metrics every X iterations
        "verbosity": 1,
        "monitor": "loss",
        "monitor_mode": "none",
        "warmup_steps": 1000,               # Defines length of ramp up from 0 learning rate
        "conf_thresh_init": 0.5,            
        "conf_thresh_change_iters": 0,      # Allows slowly lowering of detection conf thresh from higher value
        "retry_count":1,

        "unfreeze_detector": 2000,          # Iteration to unfreeze detector network
        "partial_from_gt": 0,               # Iteration to start using detection predictions
        "stop_from_gt": 20000,              # When to maximize predicted detection use
        "max_use_pred": 0.5,                # Maximum predicted detection use
        "use_all_bb_pred_for_rel_loss": true,

        "use_learning_schedule": true,
        "adapt_lr": false
    },
    "arch": "PairingGraph",                 # Class name of model
    "model": {
        "detector_checkpoint": "saved/detector/checkpoint-iteration150000.pth",
        "conf_thresh": 0.5,
        "start_frozen": true,
	"use_rel_shape_feats": "corner",
        "use_detect_layer_feats": 16,       # Assumes this is from final level of detection network
        "use_2nd_detect_layer_feats": 0,    # Specify conv after pool
        "use_2nd_detect_scale_feats": 2,    # Scale (from pools)
        "use_2nd_detect_feats_size": 64,
        "use_fixed_masks": true,
        "no_grad_feats": true,

        "expand_rel_context": 150,          # How much to pad around relationship candidates before passing to conv layers
        "featurizer_start_h": 32,           # Size ROIPooling resizes relationship crops to
        "featurizer_start_w": 32,
        "featurizer_conv": ["sep128","M","sep128","sep128","M","sep256","sep256","M",238], # Network for featurizing relationship, see below for syntax
        "featurizer_fc": null,

        "pred_nn": true,                    # Predict a new num neighbors for detections
        "pred_class": false,                # Predict a new class for detections
        "expand_bb_context": 150,           # How much to pad around detections
        "featurizer_bb_start_h": 32,        # Size ROIPooling resizes detection crops to
        "featurizer_bb_start_w": 32,
        "bb_featurizer_conv": ["sep64","M","sep64","sep64","M","sep128","sep128","M",250], # Network for featurizing detections

        "graph_config": {
            "arch": "BinaryPairReal",
            "in_channels": 256,
            "layers": ["FC256","FC256"],    # Relationship classifier
            "rel_out": 1,                   # one output, probability of true relationship
            "layers_bb": ["FC256"]          # Detection predictor
            "bb_out": 1,                    # one output, num neighbors
        }
    }
}
  ```

Config network layer syntax:

* `[int]`: Regular 3x3 convolution with specified output channels, normalization (if any), and ReLU
* `"ReLU"`
* `"drop[float]"`/`"dropout[float]"`: Dropout, if no float amount is 0.5
* `"M"`": Maxpool (2x2)
* `"R[int]"`: Residual block with specified output channels, two 3x3 convs with correct ReLU+norm ordering (expects non-acticated input)
* `"k[int]-[int]"`: conv, norm, relu. First int specifies kernel size, second specifier output channels.
* `"d[int]-[int]"`: dilated conv, norm, relu. First int specifies dilation, second specifier output channels.
* `"[h/v]d[int]-[int]"`: horizontal or vertical dilated conv, norm, relu (horizontal is 1x3 and vertical is 3x1 kernel). First int specifies dilation, second specifier output channels. 
* `"sep[int]"`: Two conv,norm,relu blocks, the first is depthwise seperated, the second is (1x1). The int is the out channels
* `"cc[str]-k[int],d[int],[hd/vd]-[int]"`: CoordConv, str is type, k int is kernel size (default 3), d is dilation size (default 1), hd makes it horizontal (kernel is height 1), vd makes it vertical, final int is out channels 
* `"FC[int]"`: Fully-connected layer with given output channels


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


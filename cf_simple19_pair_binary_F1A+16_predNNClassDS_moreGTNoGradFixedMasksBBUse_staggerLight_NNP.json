{
    "name": "Simple19_pair_binary_F1A+16_predNNClassDS_moreGTNoGradFixedMasksBBUse_staggerLightRFh_NNP",
    "cuda": true,
    "gpu": 0,
    "save_mode": "state_dict",
    "override": true,
    "super_computer":false,
    "data_loader": {
        "data_set_name": "FormsGraphPair",
        "special_dataset": "simple",
        "data_dir": "../data/forms",
        "batch_size": 1,
        "shuffle": true,
        "num_workers": 1,
        "crop_to_page":false,
        "color":false,
        "rescale_range": [0.4,0.65],
        "crop_params": {
            "crop_size":[652,1608],
	    "pad":0
        },
        "no_blanks": true,
        "swap_circle":true,
        "no_graphics":true,
        "cache_resized_images": true,
        "rotation": false,
        "only_opposite_pairs": true


    },
    "validation": {
        "shuffle": false,
        "rescale_range": [0.52,0.52],
        "crop_params": null,
        "batch_size": 1
    },

    
    "lr_scheduler_type": "none",
 
    "optimizer_type": "Adam",
    "optimizer": {
        "lr": 0.001,
        "weight_decay": 0
    },
    "loss": {
        "box": "YoloLoss",
        "edge": "sigmoid_BCE_loss",
        "nn": "MSE",
        "class": "sigmoid_BCE_loss"
    },
    "loss_weights": {
        "box": 0.8,
        "edge": 0.8,
        "nn": 0.6,
        "class": 0.6
    },
    "loss_params": 
        {
            "box": {"ignore_thresh": 0.5,
                    "bad_conf_weight": 20.0,
                    "multiclass":true}
        },
    "metrics": [],
    "trainer": {
        "class": "GraphPairTrainer",
        "iterations": 60000,
        "save_dir": "saved/",
        "val_step": 5000,
        "save_step": 5000,
        "save_step_minor": 250,
        "log_step": 250,
        "verbosity": 1,
        "monitor": "loss",
        "monitor_mode": "none",
        "warmup_steps": 1000,
        "conf_thresh_init": 0.9,
        "conf_thresh_change_iters": 5000,
        "retry_count":2,

        "unfreeze_detector": 2000,
        "partial_from_gt": 0,
        "stop_from_gt": 20000,
        "max_use_pred": 0.5,
        "use_all_bb_pred_for_rel_loss": true,

        "use_learning_schedule": true,
        "adapt_lr": false
    },
    "arch": "PairingGraph", 
    "model": {
        "detector_checkpoint": "saved/Forms18_augRFh_staggerLight_NN/checkpoint-iteration200000.pth.tar",
        "conf_thresh": 0.5,
        "start_frozen": true,
	"use_rel_shape_feats": "corner",
        "use_detect_layer_feats": 16,
        "use_2nd_detect_layer_feats": 0,
        "use_2nd_detect_scale_feats": 1,
        "use_2nd_detect_feats_size": 32,
        "use_fixed_masks": true,
        "no_grad_feats": true,

        "expand_rel_context": 150,
        "featurizer_start_h": 32,
        "featurizer_start_w": 32,
        "featurizer_conv": ["sep128","M","sep128","sep128","M","sep256","sep256","M",238],
        "featurizer_fc": null,

        "pred_nn": true,
        "pred_class": true,
        "expand_bb_context": 150,
        "featurizer_bb_start_h": 32,
        "featurizer_bb_start_w": 32,
        "bb_featurizer_conv": ["sep64","M","sep64","sep64","M","sep128","sep128","M",238],

        "graph_config": {
            "arch": "BinaryPairReal",
            "in_channels": 256,
            "bb_out": 3,
            "rel_out": 1,
            "layers": ["FC256","FC256"],
            "layers_bb": ["FC256"]
        }
    }
}

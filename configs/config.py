import os

# MODEL = 'dinov2_reg'
# MODEL = 'dinov2'
MODEL = 'megadescriptor'

DATASETS = ['StripeSpotter',
            'CatIndividualImages',
            'CowDataset',
            'Cows2021',
            'Giraffes',
            'NDD20',
            'OpenCows2020',
            'SeaStarReID2023',
            'ZindiTurtleRecall'
]

ROOT_DIR = '/home/avisund/data/wildlife_datasets/'
CONFIG_PATH = 'configs/md-configs.json'

BEST_PARAMS = {
    "FriesianCattle2015v2": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 2e-05,
            "best_epoch": 50,
            "best_val_acc": 1.0
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.01,
            "best_epoch": 50,
            "best_val_acc": 1.0
        }
    },
    "CTai": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 2e-05,
            "best_epoch": 14,
            "best_val_acc": 0.967828418230563
        },
        "linear": {
            "batch_size": 128,
            "learning_rate": 0.002,
            "best_epoch": 2,
            "best_val_acc": 0.967828418230563
        }
    },
    "CZoo": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "best_epoch": 1,
            "best_val_acc": 0.9940828402366864
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 1e-05,
            "best_epoch": 42,
            "best_val_acc": 0.9940828402366864
        }
    },
    "DogFaceNet": {
        "attentive": {
            "batch_size": 64,
            "learning_rate": 0.0002,
            "best_epoch": 10,
            "best_val_acc": 0.5286984640258691
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.0005,
            "best_epoch": 23,
            "best_val_acc": 0.5699272433306386
        }
    },
    "FriesianCattle2017": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 1e-05,
            "best_epoch": 50,
            "best_val_acc": 0.9862068965517241
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 2e-05,
            "best_epoch": 50,
            "best_val_acc": 0.9862068965517241
        }
    },
    "IPanda50": {
        "attentive": {
            "batch_size": 128,
            "learning_rate": 0.0002,
            "best_epoch": 3,
            "best_val_acc": 0.9700272479564033
        },
        "linear": {
            "batch_size": 128,
            "learning_rate": 0.005,
            "best_epoch": 2,
            "best_val_acc": 0.9709355131698456
        }
    },
    "MacaqueFaces": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 0.002,
            "best_epoch": 1,
            "best_val_acc": 0.9960238568588469
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "best_epoch": 1,
            "best_val_acc": 0.9960238568588469
        }
    },
    "NyalaData": {
        "attentive": {
            "batch_size": 32,
            "learning_rate": 0.0002,
            "best_epoch": 5,
            "best_val_acc": 0.75
        },
        "linear": {
            "batch_size": 64,
            "learning_rate": 0.002,
            "best_epoch": 13,
            "best_val_acc": 0.7601351351351351
        }
    },
    "SeaTurtleIDHeads": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 2e-05,
            "best_epoch": 15,
            "best_val_acc": 0.9642857142857143
        },
        "linear": {
            "batch_size": 16,
            "learning_rate": 0.002,
            "best_epoch": 2,
            "best_val_acc": 0.9659468438538206
        }
    },
    "CatIndividualImages": {
        "attentive": {
            "batch_size": 16,
            "learning_rate": 0.0002,
            "best_epoch": 6,
            "best_val_acc": 0.8859607091518926
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.0005,
            "best_epoch": 11,
            "best_val_acc": 0.8979396262577863
        }
    },
    "CowDataset": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 1e-05,
            "best_epoch": 50,
            "best_val_acc": 0.9957983193277311
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.0005,
            "best_epoch": 50,
            "best_val_acc": 0.9957983193277311
        }
    },
    "Cows2021": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 1e-05,
            "best_epoch": 28,
            "best_val_acc": 0.9992790194664743
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 2e-05,
            "best_epoch": 50,
            "best_val_acc": 0.9992790194664743
        }
    },
    "Giraffes": {
        "attentive": {
            "batch_size": 32,
            "learning_rate": 0.0002,
            "best_epoch": 36,
            "best_val_acc": 0.958139534883721
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.002,
            "best_epoch": 50,
            "best_val_acc": 0.9534883720930233
        }
    },
    "NDD20": {
        "attentive": {
            "batch_size": 16,
            "learning_rate": 0.0001,
            "best_epoch": 9,
            "best_val_acc": 0.8474178403755869
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.0005,
            "best_epoch": 9,
            "best_val_acc": 0.8568075117370892
        }
    },
    "OpenCows2020": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 1e-05,
            "best_epoch": 35,
            "best_val_acc": 1.0
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 1e-05,
            "best_epoch": 50,
            "best_val_acc": 1.0
        }
    },
    "SeaStarReID2023": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 0.0002,
            "best_epoch": 7,
            "best_val_acc": 0.8547008547008547
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.001,
            "best_epoch": 38,
            "best_val_acc": 0.8803418803418803
        }
    },
    "ZindiTurtleRecall": {
        "attentive": {
            "batch_size": 8,
            "learning_rate": 0.0002,
            "best_epoch": 4,
            "best_val_acc": 0.6261530113944656
        },
        "linear": {
            "batch_size": 8,
            "learning_rate": 0.0002,
            "best_epoch": 19,
            "best_val_acc": 0.7021161150298426
        }
    },
    "StripeSpotter": {
        "attentive": {
            "batch_size": 128,
            "learning_rate": 0.01,
            "best_epoch": 14, 
            "best_val_acc": 0.9770992366412213
        },
        "linear": {
            "batch_size": 16,
            "learning_rate": 0.0005,
            "best_epoch": 3, 
            "best_val_acc": 0.9694656488549618
        }
    }
}

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'
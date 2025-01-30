import os

# MODEL = 'dinov2_reg'
# MODEL = 'dinov2'
# MODEL = 'megadescriptor'
MODEL = 'SwinT'

# DATASETS = ['FriesianCattle2015v2', 
#             'CTai',
#             'CZoo',
#             'DogFaceNet', 
#             'FriesianCattle2017',
#             'IPanda50',
#             'MacaqueFaces',
#             'NyalaData', 
#             'SeaTurtleIDHeads',
#             'StripeSpotter',
#             'CatIndividualImages',
#             'CowDataset',
#             'Cows2021',
#             'Giraffes',
#             'NDD20',
#             'OpenCows2020',
#             'SeaStarReID2023',
#             'ZindiTurtleRecall'
# ]

DATASETS = [
    'AmvrakikosTurtles',
    'ReunionTurtles',
    'SouthernProvinceTurtles',
    'SeaStarReID2023',
    'SeaTurtleIDHeads',
    'ZindiTurtleRecall'
]

ROOT_DIR = '/home/avisund/data/wildlife_datasets/'
CONFIG_PATH = 'configs/md-configs.json'
# CONFIG_PATH = 'configs/dinov2-configs-2.json'

BEST_PARAMS = {
    "FriesianCattle2015v2": {
        "megadescriptor": {
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
        }
    },
    "CTai": {
        "megadescriptor": {
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
        }
    },
    "CZoo": {
        "megadescriptor": {
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
        }
    },
    "DogFaceNet": {
        "megadescriptor": {
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
        }
    },
    "FriesianCattle2017": {
        "megadescriptor": {
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
        }
    },
    "IPanda50": {
        "megadescriptor": {
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
        }
    },
    "MacaqueFaces": {
        "megadescriptor": {
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
        }
    },
    "NyalaData": {
        "megadescriptor": {
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
        }
    },
    "SeaTurtleIDHeads": {
        "megadescriptor": {
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
            },
            "weighted": {
                "batch_size": 8,
                "learning_rate": 0.0005,
                "best_epoch": 6,
                "best_val_acc": 0.9659468438538206
            },
            "gem": {
                "batch_size": 8,
                "learning_rate": 0.0001,
                "best_epoch": 14,
                "best_val_acc": 0.9651162790697675
            }
        }, 
        "SwinT": {
            "linear": {
                "batch_size": 16,
                "learning_rate": 0.002,
                "best_epoch": 45,
                "best_val_acc": 0.4883720930232558
            }, 
            "gem": {
                "batch_size": 64,
                "learning_rate": 0.02,
                "best_epoch": 45,
                "best_val_acc": 0.43272425249169433
            }, 
            "weighted": {
                "batch_size": 128,
                "learning_rate": 0.0005,
                "best_epoch": 50,
                "best_val_acc": 0.45016611295681064
            },
            "attentive": {
                "batch_size": 128,
                "learning_rate": 0.0005,
                "best_epoch": 10,
                "best_val_acc": 0.5041528239202658
            }
        }
    },
    "CatIndividualImages": {
        "megadescriptor": {
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
        }
    },
    "CowDataset": {
        "megadescriptor": {
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
        }
    },
    "Cows2021": {
        "megadescriptor": {
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
        }
    },
    "Giraffes": {
        "megadescriptor": {
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
        }
    },
    "NDD20": {
        "megadescriptor": {
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
        }
    },
    "OpenCows2020": {
        "megadescriptor": {
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
        }
    },
    "ZindiTurtleRecall": {
        "megadescriptor": {
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
            },
            "weighted": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "best_epoch": 3,
                "best_val_acc": 0.6858383071079761
            },
            "gem": {
                "batch_size": 8,
                "learning_rate": 0.002,
                "best_epoch": 3,
                "best_val_acc": 0.7015735214324471
            }
        }, 
        "SwinT": {
            "linear": {
                "batch_size": 128,
                "learning_rate": 0.005,
                "best_epoch": 15,
                "best_val_acc": 0.08193163320672817
            }, 
            "gem": {
                "batch_size": 128,
                "learning_rate": 0.0005,
                "best_epoch": 27,
                "best_val_acc": 0.033640803038524146
            }, 
            "weighted": {
                "batch_size": 128,
                "learning_rate": 0.0005,
                "best_epoch": 6,
                "best_val_acc": 0.08790016277807922
            },
            "attentive": {
                "batch_size": 128,
                "learning_rate": 0.0002,
                "best_epoch": 6,
                "best_val_acc": 0.09441128594682582
            }
        }
    },
    "StripeSpotter": {
        "megadescriptor": {
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
    },
    "SeaStarReID2023": {
        "megadescriptor": {
            "attentive": {
                "batch_size": 8,
                "learning_rate": 0.0002,
                "best_epoch": 5,
                "best_val_acc": 0.8433048433048433
            },
            "linear": {
                "batch_size": 64,
                "learning_rate": 0.01,
                "best_epoch": 26,
                "best_val_acc": 0.886039886039886
            },
            "weighted": {
                "batch_size": 16,
                "learning_rate": 0.02,
                "best_epoch": 6,
                "best_val_acc": 0.8433048433048433
            },
            "gem": {
                "batch_size": 16,
                "learning_rate": 0.002,
                "best_epoch": 50,
                "best_val_acc": 0.8717948717948718
            }
        }, 
        "SwinT": {
            "linear": {
                "batch_size": 128,
                "learning_rate": 0.02,
                "best_epoch": 50,
                "best_val_acc": 0.6809116809116809
            }, 
            "gem": {
                "batch_size": 128,
                "learning_rate": 0.05,
                "best_epoch": 50,
                "best_val_acc": 0.6267806267806267
            }, 
            "weighted": {
                "batch_size": 16,
                "learning_rate": 0.005,
                "best_epoch": 12,
                "best_val_acc": 0.6638176638176638
            },
            "attentive": {
                "batch_size": 16,
                "learning_rate": 0.0002,
                "best_epoch": 16,
                "best_val_acc": 0.7264957264957265
            }
        }
    },
    "AmvrakikosTurtles": {
        "megadescriptor": {
            "attentive": {
                "batch_size": 128,
                "learning_rate": 0.0005,
                "best_epoch": 6,
                "best_val_acc": 0.4444444444444444
            },
            "linear": {
                "batch_size": 64,
                "learning_rate": 0.05,
                "best_epoch": 1,
                "best_val_acc": 0.4074074074074074
            },
            "weighted": {
                "batch_size": 128,
                "learning_rate": 0.005,
                "best_epoch": 3,
                "best_val_acc": 0.4074074074074074
            },
            "gem": {
                "batch_size": 8,
                "learning_rate": 0.05,
                "best_epoch": 3,
                "best_val_acc": 0.3333333333333333
            }
        },
        "SwinT": {
            "linear": {
                "batch_size": 8,
                "learning_rate": 0.02,
                "best_epoch": 7,
                "best_val_acc": 0.1111111111111111
            }, 
            "gem": {
                "batch_size": 32,
                "learning_rate": 0.02,
                "best_epoch": 3,
                "best_val_acc": 0.1111111111111111
            }, 
            "weighted": {
                "batch_size": 32,
                "learning_rate": 0.02,
                "best_epoch": 3,
                "best_val_acc": 0.18518518518518517
            },
            "attentive": {
                "batch_size": 16,
                "learning_rate": 0.005,
                "best_epoch": 19,
                "best_val_acc": 0.1111111111111111
            }
        }
    },
    "ReunionTurtles": {
        "megadescriptor": {
            "attentive": {
                "batch_size": 16,
                "learning_rate": 0.0002,
                "best_epoch": 5,
                "best_val_acc": 0.6
            },
            "linear": {
                "batch_size": 32,
                "learning_rate": 0.01,
                "best_epoch": 2,
                "best_val_acc": 0.4888888888888889
            },
            "weighted": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "best_epoch": 50,
                "best_val_acc": 0.5111111111111111
            },
            "gem": {
                "batch_size": 16,
                "learning_rate": 0.05,
                "best_epoch": 3,
                "best_val_acc": 0.4666666666666667
            }
        }, 
        "SwinT": {
            "linear": {
                "batch_size": 32,
                "learning_rate": 0.05,
                "best_epoch": 22,
                "best_val_acc": 0.24444444444444444
            }, 
            "gem": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "best_epoch": 17,
                "best_val_acc": 0.15555555555555556
            }, 
            "weighted": {
                "batch_size": 16,
                "learning_rate": 0.01,
                "best_epoch": 8,
                "best_val_acc": 0.13333333333333333
            },
            "attentive": {
                "batch_size": 8,
                "learning_rate": 0.0002,
                "best_epoch": 1,
                "best_val_acc": 0.15555555555555556
            }
        }
    },
    "SouthernProvinceTurtles": {
        "megadescriptor": {
            "attentive": {
                "batch_size": 8,
                "learning_rate": 5e-05,
                "best_epoch": 25,
                "best_val_acc": 0.8133333333333334
            },
            "linear": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "best_epoch": 50,
                "best_val_acc": 0.8266666666666667
            },
            "weighted": {
                "batch_size": 8,
                "learning_rate": 0.0002,
                "best_epoch": 50,
                "best_val_acc": 0.8133333333333334
            },
            "gem": {
                "batch_size": 8,
                "learning_rate": 0.01,
                "best_epoch": 31,
                "best_val_acc": 0.7866666666666666
            }
        }, 
        "SwinT": {
            "linear": {
                "batch_size": 32,
                "learning_rate": 0.02,
                "best_epoch": 14,
                "best_val_acc": 0.7333333333333333
            }, 
            "gem": {
                "batch_size": 128,
                "learning_rate": 0.02,
                "best_epoch": 44,
                "best_val_acc": 0.72
            }, 
            "weighted": {
                "batch_size": 32,
                "learning_rate": 0.0005,
                "best_epoch": 50,
                "best_val_acc": 0.76
            },
            "attentive": {
                "batch_size": 16,
                "learning_rate": 0.0001,
                "best_epoch": 14,
                "best_val_acc": 0.7733333333333333
            }
        }
    }
}

def get_dataset_root(dataset_name):
    return os.path.join(ROOT_DIR, dataset_name) + '/'
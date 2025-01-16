import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the log file content
log_file = """
2025-01-14 22:36:14,571 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 1e-05, batch size: 8
2025-01-14 22:36:38,088 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9651162790697675.
2025-01-14 22:36:38,089 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 2e-05, batch size: 8
2025-01-14 22:37:01,520 - INFO - Training stopped at epoch 45 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:37:01,520 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 5e-05, batch size: 8
2025-01-14 22:37:16,397 - INFO - Training stopped at epoch 27 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:37:16,398 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0001, batch size: 8
2025-01-14 22:37:28,514 - INFO - Training stopped at epoch 21 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:37:28,514 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0002, batch size: 8
2025-01-14 22:37:38,899 - INFO - Training stopped at epoch 17 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:37:38,899 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0005, batch size: 8
2025-01-14 22:37:47,280 - INFO - Training stopped at epoch 13 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9642857142857143.
2025-01-14 22:37:47,280 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.001, batch size: 8
2025-01-14 22:37:53,789 - INFO - Training stopped at epoch 9 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9626245847176079.
2025-01-14 22:37:53,789 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.002, batch size: 8
2025-01-14 22:38:02,161 - INFO - Training stopped at epoch 13 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9642857142857143.
2025-01-14 22:38:02,161 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.005, batch size: 8
2025-01-14 22:38:25,683 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9593023255813954.
2025-01-14 22:38:25,683 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.01, batch size: 8
2025-01-14 22:38:28,992 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9435215946843853.
2025-01-14 22:38:28,992 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.02, batch size: 8
2025-01-14 22:38:31,827 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9393687707641196.
2025-01-14 22:38:31,827 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.05, batch size: 8
2025-01-14 22:38:34,725 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9385382059800664.
2025-01-14 22:38:34,725 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.1, batch size: 8
2025-01-14 22:38:37,641 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9401993355481728.
2025-01-14 22:38:37,641 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 1e-05, batch size: 16
2025-01-14 22:38:49,724 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9609634551495017.
2025-01-14 22:38:49,724 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 2e-05, batch size: 16
2025-01-14 22:39:01,810 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:39:01,810 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 5e-05, batch size: 16
2025-01-14 22:39:13,944 - INFO - Training stopped at epoch 47 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9626245847176079.
2025-01-14 22:39:13,944 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0001, batch size: 16
2025-01-14 22:39:24,220 - INFO - Training stopped at epoch 38 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:39:24,220 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0002, batch size: 16
2025-01-14 22:39:32,475 - INFO - Training stopped at epoch 30 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:39:32,476 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0005, batch size: 16
2025-01-14 22:39:38,882 - INFO - Training stopped at epoch 22 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9626245847176079.
2025-01-14 22:39:38,882 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.001, batch size: 16
2025-01-14 22:39:43,654 - INFO - Training stopped at epoch 15 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9626245847176079.
2025-01-14 22:39:43,654 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.002, batch size: 16
2025-01-14 22:39:47,460 - INFO - Training stopped at epoch 11 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:39:47,460 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.005, batch size: 16
2025-01-14 22:39:49,780 - INFO - Training stopped at epoch 5 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9501661129568106.
2025-01-14 22:39:49,780 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.01, batch size: 16
2025-01-14 22:39:51,638 - INFO - Training stopped at epoch 3 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9451827242524917.
2025-01-14 22:39:51,638 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.02, batch size: 16
2025-01-14 22:39:53,264 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9401993355481728.
2025-01-14 22:39:53,264 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.05, batch size: 16
2025-01-14 22:39:54,674 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9426910299003323.
2025-01-14 22:39:54,674 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.1, batch size: 16
2025-01-14 22:39:56,090 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9410299003322259.
2025-01-14 22:39:56,090 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 1e-05, batch size: 32
2025-01-14 22:40:02,237 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9593023255813954.
2025-01-14 22:40:02,238 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 2e-05, batch size: 32
2025-01-14 22:40:08,392 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:40:08,392 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 5e-05, batch size: 32
2025-01-14 22:40:14,592 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:40:14,592 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0001, batch size: 32
2025-01-14 22:40:20,774 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:40:20,774 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0002, batch size: 32
2025-01-14 22:40:26,950 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:40:26,951 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0005, batch size: 32
2025-01-14 22:40:32,117 - INFO - Training stopped at epoch 37 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:40:32,117 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.001, batch size: 32
2025-01-14 22:40:35,909 - INFO - Training stopped at epoch 26 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9626245847176079.
2025-01-14 22:40:35,910 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.002, batch size: 32
2025-01-14 22:40:38,240 - INFO - Training stopped at epoch 14 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:40:38,240 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.005, batch size: 32
2025-01-14 22:40:44,378 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9609634551495017.
2025-01-14 22:40:44,378 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.01, batch size: 32
2025-01-14 22:40:45,244 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9476744186046512.
2025-01-14 22:40:45,244 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.02, batch size: 32
2025-01-14 22:40:46,111 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9377076411960132.
2025-01-14 22:40:46,111 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.05, batch size: 32
2025-01-14 22:40:46,861 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9493355481727574.
2025-01-14 22:40:46,861 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.1, batch size: 32
2025-01-14 22:40:47,611 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9401993355481728.
2025-01-14 22:40:47,612 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 1e-05, batch size: 64
2025-01-14 22:40:51,280 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9534883720930233.
2025-01-14 22:40:51,280 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 2e-05, batch size: 64
2025-01-14 22:40:54,956 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9601328903654485.
2025-01-14 22:40:54,956 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 5e-05, batch size: 64
2025-01-14 22:40:58,635 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9626245847176079.
2025-01-14 22:40:58,635 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0001, batch size: 64
2025-01-14 22:41:02,317 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9642857142857143.
2025-01-14 22:41:02,317 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0002, batch size: 64
2025-01-14 22:41:06,000 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:41:06,000 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0005, batch size: 64
2025-01-14 22:41:09,680 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9642857142857143.
2025-01-14 22:41:09,680 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.001, batch size: 64
2025-01-14 22:41:12,983 - INFO - Training stopped at epoch 40 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9626245847176079.
2025-01-14 22:41:12,983 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.002, batch size: 64
2025-01-14 22:41:15,409 - INFO - Training stopped at epoch 28 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:41:15,409 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.005, batch size: 64
2025-01-14 22:41:18,563 - INFO - Training stopped at epoch 37 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9617940199335548.
2025-01-14 22:41:18,563 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.01, batch size: 64
2025-01-14 22:41:19,170 - INFO - Training stopped at epoch 3 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9509966777408638.
2025-01-14 22:41:19,170 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.02, batch size: 64
2025-01-14 22:41:19,702 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9451827242524917.
2025-01-14 22:41:19,702 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.05, batch size: 64
2025-01-14 22:41:20,234 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9493355481727574.
2025-01-14 22:41:20,234 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.1, batch size: 64
2025-01-14 22:41:20,765 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.946843853820598.
2025-01-14 22:41:20,765 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 1e-05, batch size: 128
2025-01-14 22:41:23,223 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9435215946843853.
2025-01-14 22:41:23,223 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 2e-05, batch size: 128
2025-01-14 22:41:25,677 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9593023255813954.
2025-01-14 22:41:25,677 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 5e-05, batch size: 128
2025-01-14 22:41:28,134 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:41:28,134 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0001, batch size: 128
2025-01-14 22:41:30,592 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9642857142857143.
2025-01-14 22:41:30,592 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0002, batch size: 128
2025-01-14 22:41:33,050 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:41:33,050 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.0005, batch size: 128
2025-01-14 22:41:35,508 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:41:35,508 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.001, batch size: 128
2025-01-14 22:41:37,964 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9642857142857143.
2025-01-14 22:41:37,964 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.002, batch size: 128
2025-01-14 22:41:40,226 - INFO - Training stopped at epoch 41 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:41:40,226 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.005, batch size: 128
2025-01-14 22:41:40,527 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9634551495016611.
2025-01-14 22:41:40,527 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.01, batch size: 128
2025-01-14 22:41:40,826 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9593023255813954.
2025-01-14 22:41:40,827 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.02, batch size: 128
2025-01-14 22:41:41,175 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9526578073089701.
2025-01-14 22:41:41,175 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.05, batch size: 128
2025-01-14 22:41:41,523 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9460132890365448.
2025-01-14 22:41:41,523 - INFO - Grid eval for classifiers: Running experiment with dataset: SeaTurtleIDHeads, model: megadescriptor, pooling method: linear, use_class: False, learning rate: 0.1, batch size: 128
2025-01-14 22:41:41,919 - INFO - Training stopped at epoch 3 for config: {'pooling_method': 'linear', 'use_class': False}. Validation accuracy: 0.9509966777408638.
"""

# Define regex patterns
config_pattern = r"learning rate: (?P<learning_rate>[\de.-]+), batch size: (?P<batch_size>\d+)"
accuracy_pattern = r"Validation accuracy: (?P<validation_accuracy>[0-9.]+)"

dataset_regex = r"Running experiment with dataset: (\w+),"
pooling_method_regex = r"pooling method: (\w+)"

dataset_match = re.search(dataset_regex, log_file)
pooling_method_match = re.search(pooling_method_regex, log_file)

dataset_name = dataset_match.group(1) if dataset_match else "Unknown Dataset"
pooling_method = pooling_method_match.group(1) if pooling_method_match else "Unknown Pooling Method"

# Parse the log file
configs = []
accuracies = []

for line in log_file.strip().split('\n'):
    config_match = re.search(config_pattern, line)
    if config_match:
        configs.append(config_match.groupdict())
    accuracy_match = re.search(accuracy_pattern, line)
    if accuracy_match:
        accuracies.append(accuracy_match.groupdict())

# Combine data into a single structure
data = []
for config, accuracy in zip(configs, accuracies):
    data.append({
        "learning_rate": float(config["learning_rate"]),
        "batch_size": int(config["batch_size"]),
        "validation_accuracy": float(accuracy["validation_accuracy"][:-1])
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

pivot_table = df.pivot(index="batch_size", columns="learning_rate", values="validation_accuracy")

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="magma", cbar_kws={"label": "Validation Accuracy"})

heatmap_title = f"Val. Acc. Heatmap: {dataset_name}, {pooling_method} pooling."
plt.title(heatmap_title, fontsize=14)
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'plots/{dataset_name}_{pooling_method}_hyperparam_sensitivity.png')

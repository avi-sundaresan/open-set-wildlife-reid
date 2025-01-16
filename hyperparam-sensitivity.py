import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the log file content
log_file = """
2025-01-15 23:23:37,372 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 1e-05, batch size: 8
2025-01-15 23:23:44,610 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.8.
2025-01-15 23:23:44,610 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 2e-05, batch size: 8
2025-01-15 23:23:51,870 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7733333333333333.
2025-01-15 23:23:51,870 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 5e-05, batch size: 8
2025-01-15 23:23:56,134 - INFO - Training stopped at epoch 24 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7733333333333333.
2025-01-15 23:23:56,134 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0001, batch size: 8
2025-01-15 23:23:58,377 - INFO - Training stopped at epoch 10 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.76.
2025-01-15 23:23:58,377 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0002, batch size: 8
2025-01-15 23:23:59,895 - INFO - Training stopped at epoch 5 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7866666666666666.
2025-01-15 23:23:59,895 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0005, batch size: 8
2025-01-15 23:24:00,980 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7066666666666667.
2025-01-15 23:24:00,980 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.001, batch size: 8
2025-01-15 23:24:02,067 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.6266666666666667.
2025-01-15 23:24:02,067 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.002, batch size: 8
2025-01-15 23:24:03,010 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.52.
2025-01-15 23:24:03,010 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.005, batch size: 8
2025-01-15 23:24:03,951 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4.
2025-01-15 23:24:03,951 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.01, batch size: 8
2025-01-15 23:24:04,889 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.41333333333333333.
2025-01-15 23:24:04,889 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.02, batch size: 8
2025-01-15 23:24:05,826 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.5066666666666667.
2025-01-15 23:24:05,827 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.05, batch size: 8
2025-01-15 23:24:06,763 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.3466666666666667.
2025-01-15 23:24:06,764 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.1, batch size: 8
2025-01-15 23:24:07,701 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4.
2025-01-15 23:24:07,701 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 1e-05, batch size: 16
2025-01-15 23:24:13,386 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.72.
2025-01-15 23:24:13,386 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 2e-05, batch size: 16
2025-01-15 23:24:19,086 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.76.
2025-01-15 23:24:19,086 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 5e-05, batch size: 16
2025-01-15 23:24:22,761 - INFO - Training stopped at epoch 27 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7733333333333333.
2025-01-15 23:24:22,761 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0001, batch size: 16
2025-01-15 23:24:24,976 - INFO - Training stopped at epoch 14 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.76.
2025-01-15 23:24:24,976 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0002, batch size: 16
2025-01-15 23:24:26,852 - INFO - Training stopped at epoch 11 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7866666666666666.
2025-01-15 23:24:26,852 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0005, batch size: 16
2025-01-15 23:24:27,713 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7333333333333333.
2025-01-15 23:24:27,713 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.001, batch size: 16
2025-01-15 23:24:28,688 - INFO - Training stopped at epoch 3 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.68.
2025-01-15 23:24:28,689 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.002, batch size: 16
2025-01-15 23:24:29,552 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.6666666666666666.
2025-01-15 23:24:29,552 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.005, batch size: 16
2025-01-15 23:24:30,410 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.41333333333333333.
2025-01-15 23:24:30,410 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.01, batch size: 16
2025-01-15 23:24:31,153 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4.
2025-01-15 23:24:31,154 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.02, batch size: 16
2025-01-15 23:24:31,897 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4533333333333333.
2025-01-15 23:24:31,897 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.05, batch size: 16
2025-01-15 23:24:32,641 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.44.
2025-01-15 23:24:32,641 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.1, batch size: 16
2025-01-15 23:24:33,384 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.38666666666666666.
2025-01-15 23:24:33,384 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 1e-05, batch size: 32
2025-01-15 23:24:38,363 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.68.
2025-01-15 23:24:38,363 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 2e-05, batch size: 32
2025-01-15 23:24:43,346 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7733333333333333.
2025-01-15 23:24:43,346 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 5e-05, batch size: 32
2025-01-15 23:24:47,446 - INFO - Training stopped at epoch 36 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.8.
2025-01-15 23:24:47,446 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0001, batch size: 32
2025-01-15 23:24:49,880 - INFO - Training stopped at epoch 19 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7866666666666666.
2025-01-15 23:24:49,880 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0002, batch size: 32
2025-01-15 23:24:51,232 - INFO - Training stopped at epoch 8 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7733333333333333.
2025-01-15 23:24:51,232 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0005, batch size: 32
2025-01-15 23:24:51,994 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7866666666666666.
2025-01-15 23:24:51,994 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.001, batch size: 32
2025-01-15 23:24:52,756 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7466666666666667.
2025-01-15 23:24:52,756 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.002, batch size: 32
2025-01-15 23:24:53,518 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.64.
2025-01-15 23:24:53,518 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.005, batch size: 32
2025-01-15 23:24:54,179 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4666666666666667.
2025-01-15 23:24:54,179 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.01, batch size: 32
2025-01-15 23:24:54,934 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.36.
2025-01-15 23:24:54,934 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.02, batch size: 32
2025-01-15 23:24:55,590 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.52.
2025-01-15 23:24:55,590 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.05, batch size: 32
2025-01-15 23:24:56,244 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4266666666666667.
2025-01-15 23:24:56,244 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.1, batch size: 32
2025-01-15 23:24:56,897 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4666666666666667.
2025-01-15 23:24:56,897 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 1e-05, batch size: 64
2025-01-15 23:25:01,547 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.5733333333333334.
2025-01-15 23:25:01,547 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 2e-05, batch size: 64
2025-01-15 23:25:06,195 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7466666666666667.
2025-01-15 23:25:06,195 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 5e-05, batch size: 64
2025-01-15 23:25:10,839 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.76.
2025-01-15 23:25:10,839 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0001, batch size: 64
2025-01-15 23:25:13,293 - INFO - Training stopped at epoch 21 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7733333333333333.
2025-01-15 23:25:13,293 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0002, batch size: 64
2025-01-15 23:25:14,465 - INFO - Training stopped at epoch 7 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7733333333333333.
2025-01-15 23:25:14,465 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0005, batch size: 64
2025-01-15 23:25:15,266 - INFO - Training stopped at epoch 3 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7466666666666667.
2025-01-15 23:25:15,266 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.001, batch size: 64
2025-01-15 23:25:15,976 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.72.
2025-01-15 23:25:15,977 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.002, batch size: 64
2025-01-15 23:25:16,777 - INFO - Training stopped at epoch 3 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7066666666666667.
2025-01-15 23:25:16,777 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.005, batch size: 64
2025-01-15 23:25:17,762 - INFO - Training stopped at epoch 5 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.6133333333333333.
2025-01-15 23:25:17,763 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.01, batch size: 64
2025-01-15 23:25:18,750 - INFO - Training stopped at epoch 5 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.49333333333333335.
2025-01-15 23:25:18,750 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.02, batch size: 64
2025-01-15 23:25:19,457 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.37333333333333335.
2025-01-15 23:25:19,457 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.05, batch size: 64
2025-01-15 23:25:20,070 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.36.
2025-01-15 23:25:20,070 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.1, batch size: 64
2025-01-15 23:25:20,956 - INFO - Training stopped at epoch 4 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.6.
2025-01-15 23:25:20,956 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 1e-05, batch size: 128
2025-01-15 23:25:25,357 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4533333333333333.
2025-01-15 23:25:25,357 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 2e-05, batch size: 128
2025-01-15 23:25:29,754 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.64.
2025-01-15 23:25:29,754 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 5e-05, batch size: 128
2025-01-15 23:25:34,147 - INFO - Training stopped at epoch 50 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.8133333333333334.
2025-01-15 23:25:34,147 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0001, batch size: 128
2025-01-15 23:25:38,459 - INFO - Training stopped at epoch 44 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.8133333333333334.
2025-01-15 23:25:38,459 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0002, batch size: 128
2025-01-15 23:25:41,386 - INFO - Training stopped at epoch 28 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.7466666666666667.
2025-01-15 23:25:41,386 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.0005, batch size: 128
2025-01-15 23:25:42,319 - INFO - Training stopped at epoch 5 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.76.
2025-01-15 23:25:42,319 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.001, batch size: 128
2025-01-15 23:25:43,084 - INFO - Training stopped at epoch 3 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.76.
2025-01-15 23:25:43,084 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.002, batch size: 128
2025-01-15 23:25:43,759 - INFO - Training stopped at epoch 2 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.6666666666666666.
2025-01-15 23:25:43,759 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.005, batch size: 128
2025-01-15 23:25:44,350 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.44.
2025-01-15 23:25:44,350 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.01, batch size: 128
2025-01-15 23:25:44,937 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.37333333333333335.
2025-01-15 23:25:44,937 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.02, batch size: 128
2025-01-15 23:25:45,526 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.3333333333333333.
2025-01-15 23:25:45,526 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.05, batch size: 128
2025-01-15 23:25:46,114 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.32.
2025-01-15 23:25:46,114 - INFO - Grid eval for classifiers: Running experiment with dataset: SouthernProvinceTurtles, model: megadescriptor, pooling method: attentive, use_class: False, learning rate: 0.1, batch size: 128
2025-01-15 23:25:46,699 - INFO - Training stopped at epoch 1 for config: {'pooling_method': 'attentive', 'use_class': False}.  Validation accuracy: 0.4666666666666667.
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

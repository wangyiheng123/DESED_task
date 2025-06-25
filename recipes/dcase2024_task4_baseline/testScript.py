import os

import numpy as np
import pandas as pd
import psds_eval
import sed_eval
import sed_scores_eval
from psds_eval import PSDSEval, plot_psd_roc


if __name__ == '__main__':
    ground_truth_file = "../../data/dcase/dataset/metadata/validation/validation.tsv"
    durations_file = "../../data/dcase/dataset/metadata/validation/validation_durations.tsv"
    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")
    print(gt)
    print(durations)
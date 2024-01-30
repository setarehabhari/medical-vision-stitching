import os
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

Metrics = namedtuple("Metrics", ["AUC", "ACC"])


class Evaluator:
    def __init__(self, flag, task, split, root=None):

        self.flag = flag
        self.split = split
        self.task = task

    def evaluate(self, y_true, y_score, save_folder=None, run=None):
        auc = getAUC(y_true, y_score, self.task)
        acc = getACC(y_true, y_score, self.task)
        metrics = Metrics(auc, acc)

        if save_folder is not None:
            path = os.path.join(save_folder,
                                self.get_standard_evaluation_filename(metrics, run))
            pd.DataFrame(y_score).to_csv(path, header=None)
        return metrics

    def get_standard_evaluation_filename(self, metrics, run=None):
        eval_txt = "_".join(
            [f"[{k}]{v:.3f}" for k, v in zip(metrics._fields, metrics)])

        if run is None:
            import time
            run = time.time()

        ret = f"{self.flag}_{self.split}_{eval_txt}@{run}.csv"
        return ret
    
    def save_results(self, y_true, y_score, task, outputpath, threshold = 0.5):
        '''Save ground truth and scores
        :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
        :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
        :param outputpath: path to save the result csv

        '''    
        y_true = y_true.squeeze()
        y_score = y_score.squeeze()

        if task == 'multi-label, binary-class':
            y_pre = y_score > threshold
            acc = 0
            for label in range(y_true.shape[1]):
                label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
                acc += label_acc
            ret = acc / y_true.shape[1]
        elif task == 'binary-class':
            if y_score.ndim == 2:
                y_score = y_score[:, -1]
            else:
                assert y_score.ndim == 1
        
            cm = confusion_matrix(y_true, y_score > threshold)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(outputpath + "confusion_matrix.jpg")

        else:
            cm = confusion_matrix(y_true, np.argmax(y_score, axis=-1))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(outputpath + "confusion_matrix.jpg")


    @classmethod
    def parse_and_evaluate(cls, path, run=None):
        '''Parse and evaluate a standard result file.
        
        A standard result file is named as:
            {flag}_{split}|*|.csv (|*| means anything)

        A standard evaluation file is named as:
            {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv

        In result/evaluation file, each line is (dataset index,float prediction).

        For instance,
        octmnist_test_[AUC]0.672_[ACC]0.892@3.csv
            0,0.125,0.275,0.5,0.2
            1,0.5,0.125,0.275,0.2
        '''
        folder, filename = os.path.split(path)
        flag, split_, *_ = filename.split("_")
        if split_.startswith('train'):
            split = "train"
        elif split_.startswith('val'):
            split = "val"
        elif split_.startswith('test'):
            split = "test"
        else:
            raise ValueError

        if run is None:
            assert "@" in filename
            run = filename.split("@")[-1].split(".")[0]

        evaluator = cls(flag, split)

        df = pd.read_csv(path, index_col=0, header=None)
        y_score = df.sort_index().values

        metrics = evaluator.evaluate(y_score, folder, run)
        print(metrics)

        return metrics
    
    
    

def getAUC(y_true, y_score, task):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == 'multi-label, binary-class':
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == 'multi-label, binary-class':
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == 'binary-class':
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1

        ret = accuracy_score(y_true, y_score > threshold)
        
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret
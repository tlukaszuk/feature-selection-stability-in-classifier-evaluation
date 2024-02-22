from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import numpy as np
import random
from abc import ABC, abstractmethod


class BaseTrainsPDiff(ABC):

    def __init__(self, train_size, p, random_state=None):
        self.train_size = train_size
        self.p = p
        self.random_state = random_state

    
    @abstractmethod
    def split(self, X, y):
        pass



class TrainsPDiff(BaseTrainsPDiff):
    """
    Split cross-validation.
    Provides train/test indices to split data in train/test sets.
    In the initial step it splits indices on train (with size train_size + p) and test (the rest).
    Then, using the split function, it returns the train_size indexes as the training set
    and the postponed test indices + p as the test set.
    Guarantees the same size of training part for each split.
    The training parts of any two splits differ exactly p indices.

    Attributes
    ----------
    train_size : int
        Size of the training part of the split.
    p : int
        The number of objects by which the training parts of any two splits differ.
    random_state : int, default=None
        Controls the randomness of the training and testing indices produced.

    Example
    -------
    tpd = TrainsPDiff(50, 5, 0)

    for train_index, test_index in tpd.split(X, y):
        print(train_index, test_index)
    """

    def split(self, X, y):
        if self.train_size + self.p > len(X):
            raise ValueError(f"The number of objects in X is too small, for given parameters X should contain at least {self.train_size + self.p} objects.")
        if self.train_size + self.p < len(X):
            sss = ShuffleSplit(n_splits=1, train_size=self.train_size+self.p, random_state=self.random_state)
            sss_train_index, sss_test_index = list(sss.split(X, y))[0]
        else:
            sss_train_index, sss_test_index = np.arange(len(X)), np.arange(0)
        base_index = [i for i in sss_train_index]
        random.seed(self.random_state)
        while len(base_index) >= self.p:
            test_index = random.sample(base_index, self.p)
            train_index = np.array([i for i in sss_train_index if i not in test_index])
            base_index = [i for i in base_index if i not in test_index]
            test_index = np.append(test_index, sss_test_index)
            yield train_index, test_index



class StratifiedTrainsPDiff(BaseTrainsPDiff):
    """
    Stratified split cross-validation.
    Provides train/test indices to split data in train/test sets.
    In the initial step it stratified splits indices on train (with size train_size + p) and test (the rest).
    Then, using the split function, it returns the train_size indexes as the training set
    and the postponed test indices + p as the test set.
    Guarantees the same size of training part for each split.
    The training parts of any two splits differ exactly p indices.

    Attributes
    ----------
    train_size : int
        Size of the training part of the split.
    p : int
        The number of objects by which the training parts of any two splits differ.
    random_state : int, default=None
        Controls the randomness of the training and testing indices produced.

    Example
    -------
    stpd = StratifiedTrainsPDiff(50, 5, 0)

    for train_index, test_index in stpd.split(X, y):
        print(train_index, test_index)
    """
        
    def split(self, X, y):
        labels = np.unique(y)
        if self.train_size + self.p > len(X):
            raise ValueError(f"The number of objects in X is too small, for given parameters X should contain at least {self.train_size + self.p} objects.")
        if self.train_size + self.p + len(labels) < len(X):
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.train_size+self.p, random_state=self.random_state)
            sss_train_index, sss_test_index = list(sss.split(X, y))[0]
        elif self.train_size + self.p < len(X):
            sss = ShuffleSplit(n_splits=1, train_size=self.train_size+self.p, random_state=self.random_state)
            sss_train_index, sss_test_index = list(sss.split(X, y))[0]
        else:
            sss_train_index, sss_test_index = np.arange(len(X)), np.arange(0, dtype=int)
        base_indexes = [[i for i in sss_train_index if y[i]==label] for label in labels]
        random.seed(self.random_state)
        while True:
            len_base_indexes = sum([len(base_index) for base_index in base_indexes])
            if len_base_indexes < self.p:
                break
            p_contributions = [self.p*len(base_index)/len_base_indexes for base_index in base_indexes]
            test_indexes = [random.sample(base_index, min(round(pc+1e-5),len(base_index))) for base_index,pc in zip(base_indexes,p_contributions)]
            test_index = np.hstack([ti for ti in test_indexes if len(ti)>0])
            while len(test_index) > self.p:
                idx = random.choice(test_index)
                test_index = test_index[test_index != idx]
                for ti in test_indexes:
                    if idx in ti:
                        ti.remove(idx)
            train_index = np.array([i for i in sss_train_index if i not in test_index])
            base_indexes = [[i for i in base_index if i not in ti] for base_index,ti in zip(base_indexes,test_indexes)]
            test_index = np.append(test_index, sss_test_index)
            yield train_index, test_index

import numpy as np


class FeatureSpace:
    
    def __init__(self, dim):
        self.__universum = np.arange(dim)
        self.__features = np.zeros(0, dtype=int)
        self.__feature_state = np.full((dim), False)
        self.__feature_weight = np.full((dim), 1.0)
        self.__feature_weight[-1] = 0


    @property
    def features(self):
        return self.__features

    @property
    def feature_weight(self):
        return self.__feature_weight

    @property
    def features_outside_space(self):
        return self.__universum[~self.__feature_state]

        
    def set_full(self):
        self.__features = np.array(self.__universum)
        self.__feature_state = np.full((len(self.__universum)), True)
    
    
    def add_feature(self, index):
        if index not in self.__universum:
            raise ValueError('Feature with index [{}] not present in universum'.format(index))
        if self.__feature_state[index]:
            raise ValueError('Feature space already contains feature with index [{}]'.format(index))
        self.__features = np.append(self.__features, [index])
        self.__feature_state[index] = True
    
    
    def remove_feature(self, index):
        if not self.__feature_state[index]:
            raise ValueError('Feature space does not contain a feature with index [{}]'.format(index))
        self.__features = np.delete(self.__features, self.index(index))
        self.__feature_state[index] = False


    def features_positions(self, indicies):
        sorter = np.argsort(self.__features)
        return sorter[np.searchsorted(self.__features, indicies, sorter=sorter)]


    def feature_position(self, index):
        return np.where(self.__features == index)[0][0]
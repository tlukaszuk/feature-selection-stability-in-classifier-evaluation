from statistics import mean
import numpy as np


class StabilityMeasure():

    def _intersection_min(fs1, fs2):
        return len(fs1.intersection(fs2)) / min(len(fs1),len(fs2))

    def _intersection_max(fs1, fs2):
        return len(fs1.intersection(fs2)) / max(len(fs1),len(fs2))

    def _Lustgarten(fs1, fs2, n): #range (-1,1]
        r = len(fs1.intersection(fs2))
        k1 = len(fs1)
        k2 = len(fs2)
        if (k1==0) or (k2==0):
            return 0
        return  (r - k1*k2/n) / ( min(k1,k2) - max(0,k1+k2-n) )

    def _get_similarity_list(list_of_subsets, n, similarity_function):
        counts = {}
        results = []
        for i in range(0,len(list_of_subsets)-1):
            fs1 = list_of_subsets[i]
            for j in range(i+1,len(list_of_subsets)):
                fs2 = list_of_subsets[j]
                s = similarity_function(fs1, fs2, n)
                results.append(s)
        return results

    def _get_similarity_mean(list_of_subsets, n, similarity_function):
        return mean(StabilityMeasure._get_similarity_list(list_of_subsets, n, similarity_function))

    def _features_list_to_matrix(list_of_subsets,n):
        Z = np.zeros((len(list_of_subsets),n))
        for i,features in enumerate(list_of_subsets):
            for f in features:
                Z[i,f] = 1
        return Z  #Z = [...] Mxd numpy array    

    def Lustgarten(list_of_subsets, n):
        """
        Parameters
        ----------
        list_of_subsets : list[set]
            Selected features sets
        n : int
            Initial number of features

        Returns
        -------
        similarity_factor
            range [0,1]
        """
        return StabilityMeasure._get_similarity_mean(list_of_subsets, n, StabilityMeasure._Lustgarten)

    def Nogueira(list_of_subsets, n):
        """
        Parameters
        ----------
        list_of_subsets : list[set]
            Selected features sets
        n : int
            Initial number of features

        Returns
        -------
        similarity_factor
            range [0,1]
        """
        Z = StabilityMeasure._features_list_to_matrix(list_of_subsets, n)
        d = Z.shape[1]
        kbar = Z.sum(1).mean()
        if kbar == 0:
            return 0
        return 1. - Z.var(0, ddof=1).mean() / ((kbar/d)*(1.-kbar/d))
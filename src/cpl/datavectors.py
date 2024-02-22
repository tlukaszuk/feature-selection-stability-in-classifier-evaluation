import numpy as np

from .consts import INFINITY, ZERO, EPSILON, EQUAL_ZERO



class FeatureVectors:
    """A class aggregating all real feature vectors and their state information
    during the execution of the optimization procedure.
    """
    
    def __init__(self, X, y, sample_weight=None): 
        # weights of vectors
        if sample_weight is not None:
            self.sample_weight = np.array(sample_weight)
        else:
            weight_Cp = 0.5 / sum(y)
            weight_Cm = 0.5 / sum(~y)
            self.sample_weight = np.array([weight_Cp if label else weight_Cm for label in y])
            
        # augmented vectors
        def aug_x(x,label):
            ax = np.append(x, [-1.])
            if not label:
                ax *= -1.
            return ax
        self.vectors = np.array([aug_x(x,label) for x,label in zip(X,y)])
        
        # additional information, vector statuses, current parameter values
        self.in_base = np.full((len(self.vectors)), False)
        self.on_positive_side = np.full((len(self.vectors)), True)
        self.products_fv_B1l = np.zeros(len(self.vectors))
        self.products_fv_vertex = np.zeros(len(self.vectors))
        
        
    def specify_vectors_on_exit_edge(self, l, hold_direction, bases, idxs_tbc=None, spread_edges=False):
        """
        Parameters
        ----------
        l : int
            Index of base vector leaving the base, exit edge
        hold_direction : bool
            Direction of exit edge
        bases : Bases instance
            Object with information about the status of the primary base and inverse base
        idxs_tbc : Iterable[int], default None
            The list of indices of feature vectors to be considered as candidates to appear on the output edge
        spread_edges : bool, default False
            Whether to spread the edges crossing the exit edge, needed for degeneration
        """
        if idxs_tbc is None:
            idxs_tbc = [i for i,ib in enumerate(self.in_base) if (not ib)]
        else:
            idxs_tbc = [i for i,ib in zip(idxs_tbc, self.in_base[list(idxs_tbc)]) if (not ib)]
        self.products_fv_B1l[idxs_tbc] = bases.dots_fvs_B1(idxs_tbc, l)
        idxs_tbc = [i for i,p in zip(idxs_tbc, self.products_fv_B1l[idxs_tbc]) if not EQUAL_ZERO(p)]
        if hold_direction == False:
            self.products_fv_B1l[idxs_tbc] = -self.products_fv_B1l[idxs_tbc]
        if not spread_edges:
            idxs_tbc = [i for i,pos,p in zip(idxs_tbc, self.on_positive_side[idxs_tbc], self.products_fv_B1l[idxs_tbc])
                        if not (pos ^ (p>0))]
            distances = [(1. - pfvv) / p for pfvv,p in zip(self.products_fv_vertex[idxs_tbc], self.products_fv_B1l[idxs_tbc])]
        else:
            idxs_tbc = [i for i,pfvv,p in zip(idxs_tbc, self.products_fv_vertex[idxs_tbc], self.products_fv_B1l[idxs_tbc])
                        if not ((2. + i - pfvv > 0) ^ (p>0))]
            distances = [(2. + i - pfvv) / p for i,pfvv,p in zip(idxs_tbc, self.products_fv_vertex[idxs_tbc], self.products_fv_B1l[idxs_tbc])]
        return [(True, i, dist) for i,dist in zip(idxs_tbc, distances)]
    
    
    def update_products_fv_vertex(self, kv, idxs_tbc=None):
        if idxs_tbc is None:
            self.products_fv_vertex[self.in_base] = 1.0
            self.products_fv_vertex[~self.in_base] += kv[2] * self.products_fv_B1l[~self.in_base]
        else:
            self.products_fv_vertex[idxs_tbc] += kv[2] * self.products_fv_B1l[idxs_tbc]


    def recalculate_product_fv_vertex(self, fv_id, vertex, fs):
        self.products_fv_vertex[fv_id] = np.dot(self.vectors[fv_id][fs.features], vertex[fs.features])


    def recalculate_products_fv_vertex(self, vertex, fs):
        self.products_fv_vertex[self.in_base] = 1.0
        self.products_fv_vertex[~self.in_base] = [np.dot(fv[fs.features], vertex[fs.features]) for fv in self.vectors[~self.in_base]]

    
        
class UnitVectors:
    """A class aggregating all artificial unit vectors and their state information
    during the execution of the optimization procedure.
    """
    
    def __init__(self, dim):
        self.dim = dim
        self.ev = np.full((self.dim), 1.)
        self.in_base = np.full((self.dim), False)
        self.products_uv_B1l = np.zeros(self.dim)
        
    
    def specify_vectors_on_exit_edge(self, l, hold_direction, bases, vertex, fs, idxs_tbc=None, spread_edges=False):
        """
        Parameters
        ----------
        l : int
            Index of base vector leaving the base, exit edge
        hold_direction : bool
            Direction of exit edge
        bases : Bases instance
            Object with information about the status of the primary base and inverse base
        vertex : np.array
            Current vertex
        fs : FeatureSpace instance
            Object with information about current feature space
        idxs_tbc : Iterable[int], default None
            The list of indices of unit vectors to be considered as candidates to appear on the output edge
        spread_edges : bool, default False
            Whether to spread the edges crossing the exit edge, needed for degeneration
        """
        if idxs_tbc is None:
            idxs_tbc = [fs.features[i] for i in np.where(~self.in_base[fs.features])[0]]
        else:
            idxs_tbc = [i for i,ib in zip(idxs_tbc, self.in_base[list(idxs_tbc)]) if (not ib)]
        self.products_uv_B1l = bases.get_B1_vector(l)
        idxs_tbc = [i for i,p in zip(idxs_tbc, self.products_uv_B1l[idxs_tbc]) if not EQUAL_ZERO(p)]
        if not hold_direction:
            self.products_uv_B1l[idxs_tbc] = -self.products_uv_B1l[idxs_tbc]
        if not spread_edges:
            idxs_tbc = [i for i,ev,p in zip(idxs_tbc, self.ev[idxs_tbc], self.products_uv_B1l[idxs_tbc])
                        if not ((ev>0) ^ (p<0))]
            distances = [-v/p for v,p in zip(vertex[idxs_tbc], self.products_uv_B1l[idxs_tbc])]
        else:
            idxs_tbc = [i for i,ev,p in zip(idxs_tbc, self.ev[idxs_tbc], self.products_uv_B1l[idxs_tbc])
                        if not ((ev>0) ^ (p<0))]
            idxs_tbc = [i for i,v,p in zip(idxs_tbc, vertex[idxs_tbc], self.products_uv_B1l[idxs_tbc])
                        if not ((len(fs.features) + 1. + i - v > 0) ^ (p>0))]
            distances = [(len(fs.features) + 1. + i - v) / p for i,v,p in zip(idxs_tbc, vertex[idxs_tbc], self.products_uv_B1l[idxs_tbc])]
        return [(False, i, dist) for i,dist in zip(idxs_tbc,distances)]
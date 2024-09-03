from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
import math

from sklearn.utils.validation import check_array, check_X_y, check_is_fitted

from .featurespace import FeatureSpace
from .datavectors import FeatureVectors, UnitVectors
from .bases import Bases
from .consts import INFINITY, ZERO, EQUALS_EPSILON


class OptimizationMode(Enum):
    SEP_AREA_MODE = 1
    OPT_HYP_MODE = 2



class ProcedureState(Enum):
    NORMAL_STATE = 1
    BEGIN_DEGENERATION_STATE = 2
    DEGENERATION_STATE = 3
    END_DEGENERATION_STATE = 4



class CPLBaseClassifier(ABC):
    """
    Basics, common parts for different types of CPL classifiers
    """

    def __init__(self, C="auto", use_theta=True):
        self.__verbose = 0

        # control parameters
        self._optimization_mode = OptimizationMode.OPT_HYP_MODE
        self._C = C
        self._use_theta = use_theta


    def __repr__(self) -> str:
        return "CPLBaseClassifier"


    def get_params(self, deep=True) -> dict:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"C":self._C}


    def pretty_hyperplane(self):
        """
        Return the separating hyperplane in a more readable form, with ordered coefficient values.
        """
        check_is_fitted(self)
        hyp = sorted(
            [(f'x{f}', round(c, 4)) for f, c in zip(
                self.hyperplane_["features"], self.hyperplane_["coefs"])],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        hyp += [('theta', round(self.hyperplane_["theta"], 4))]
        return hyp


    def fit(self, X, y, sample_weight=None, verbose=0):
        """
        Build a classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values, class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
        verbose : int, default=0
            Refers to process logging, 0 means no logging, 1 means show informations during processing

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        classes_ = np.unique(y)
        if len(classes_) != 2:
            raise ValueError(
                f"y must contain exactly 2 label values, but has such values {classes_}")
        self.classes_ = classes_
        y = (y == self.classes_[1])

        self.__verbose = verbose
        self.__init_procedure(X, y, sample_weight)
        self._method_fit()
        self.__init_model()
        self.__verbose = 0
        return self


    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-likeof shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        X = check_array(X)
        predictions_bool = np.dot(X[:, self.hyperplane_['features']], np.array(self.hyperplane_['coefs'])) - self.hyperplane_['theta'] > 0
        return np.array([self.classes_[int(pb)] for pb in predictions_bool])


    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute classes_.
        """
        check_is_fitted(self)
        X = check_array(X)
        predictions_float = np.dot(X[:, self.hyperplane_['features']], np.array(self.hyperplane_['coefs'])) - self.hyperplane_['theta']
        def sigmoid(x): return 1 / (1 + math.exp(-2*x))
        vfunc = np.vectorize(sigmoid)
        proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float64)
        proba[:,1] = vfunc(predictions_float)
        proba[:,0] = 1.-proba[:,1]
        return proba


    @abstractmethod
    def _method_fit(self):
        """
        A concrete optimisation method. An abstract method that should be implemented
        in classes inheriting from CPLBaseClassifier.
        """
        pass


    def __init_model(self):
        """
        Set the parameters used in the prediction
        """
        self.hyperplane_ = {
            "features": [f for f in self.fs.features if (self.uvs.in_base[f] == False) and (not self._use_theta or f != self.dim)],
            "coefs": [self.vertex[f] for f in self.fs.features if (self.uvs.in_base[f] == False) and (not self._use_theta or f != self.dim)],
            "theta": self.vertex[self.dim] if self._use_theta and (self.dim in self.fs.features) and (self.uvs.in_base[self.dim] == False) else 0.
        }


    @property
    def feature_importances_(self):
        """
        Return the feature importances.
        The importance of a feature is computed as the (normalized) weight in decision rule brought by that feature.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized weights in classifier decision rule.
        """
        check_is_fitted(self)
        importances = np.zeros(self.dim)
        importances[self.hyperplane_["features"]] = np.abs(self.hyperplane_["coefs"])
        normalizer = np.sum(importances)
        if normalizer > 0.0:
            importances = importances / normalizer
        return importances


    def __init_procedure(self, X, y, sample_weight=None):
        self.num, self.dim = X.shape

        self.step = 0

        # feature weights
        feature_weight = np.abs(X).max(axis=0)
        n_features = self.dim
        if self._use_theta:
            n_features += 1
            feature_weight = np.append(feature_weight, [0.0])
        self.fs = FeatureSpace(n_features, feature_weight)

        # real and unit vectors
        self.fvs = FeatureVectors(X, y, sample_weight, self._use_theta)
        self.uvs = UnitVectors(n_features)
        self.bases = Bases(self.num, n_features, self.fvs, self.uvs, self.fs)
        self.vertex = np.zeros(n_features)
        self.cv = np.zeros(n_features)
        self.products_B1_cv = np.zeros(n_features)
        if self._C == "auto":
            self.p_lambda = 0.0005 / self.num
        else:
            self.p_lambda = 0.1 - self._C/10

        self.seq = []
        self.epsilon_neighbourhood = set()

        self.procedure_state = ProcedureState.NORMAL_STATE


    def _check_stop_and_init_s(self):
        self.lv = (self.bases.B_type[self.l], self.bases.B_index[self.l])
        self.hold_direction = (self.products_B1_cv[self.l] >= 0)
        if self.hold_direction == True:
            self.s = self.products_B1_cv[self.l]
        else:
            self.s = -self.products_B1_cv[self.l]
            if self.lv[0]:
                self.s -= self.fvs.sample_weight[self.lv[1]]
                self._modify_cv_minus(self.lv[1])
            else:
                self.s -= 2 * self.p_lambda * self.fs.feature_weight[self.lv[1]]
                self._modify_cv_unit(self.lv[1])
        return self.s >= ZERO/1e3


    def _modify_cv_minus(self, fv_id):
        self.fvs.on_positive_side[fv_id] = True
        self.cv[self.fs.features] += self.fvs.vectors[fv_id][self.fs.features] * self.fvs.sample_weight[fv_id]


    def _modify_cv_plus(self, fv_id):
        self.fvs.on_positive_side[fv_id] = False
        self.cv[self.fs.features] -= self.fvs.vectors[fv_id][self.fs.features] * self.fvs.sample_weight[fv_id]


    def _modify_cv_unit(self, uv_id):
        if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
            self.cv[uv_id] += 2 * self.p_lambda * self.uvs.ev[uv_id] * self.fs.feature_weight[uv_id]
        self.uvs.ev[uv_id] *= -1


    def _get_cv_projection_correction_by_fv(self, fv_id):
        return abs(self.fvs.products_fv_B1l[fv_id]) * self.fvs.sample_weight[fv_id]


    def _get_cv_projection_correction_by_uv(self, uv_id):
        return 2 * self.p_lambda * abs(self.uvs.products_uv_B1l[uv_id]) * self.fs.feature_weight[uv_id]


    def _update_products_B1_cv(self):
        self.products_B1_cv[self.fs.features] = self.bases.dots_B1s_realv(self.cv)


    def _go_to_new_vertex(self):
        # sort sequence on exit edge
        self.seq.sort(key=lambda x: x[2])

        # moving along the exit edge until the vertex where s <= 0
        for i, v in enumerate(self.seq):
            self.kv = (*v, i)
            if v[0]:
                self.s -= self._get_cv_projection_correction_by_fv(v[1])
                if self.fvs.products_fv_B1l[v[1]] > 0:
                    self._modify_cv_plus(v[1])
                elif (self.s > 0) and (i+1 < len(self.seq)):
                    self._modify_cv_minus(v[1])
                if self.s <= 0:
                    break
            else:
                self.s -= self._get_cv_projection_correction_by_uv(v[1])
                if self.s <= 0:
                    break
                self._modify_cv_unit(v[1])

        return self.s <= 0


    def _find_sequence_on_exit_edge(self):
        # determine feature vectors and unit vectors on exit edge
        if self.procedure_state == ProcedureState.NORMAL_STATE:
            self.seq = self.fvs.specify_vectors_on_exit_edge(self.l, self.hold_direction, self.bases)
            self.seq += self.uvs.specify_vectors_on_exit_edge(self.l, self.hold_direction, self.bases, self.vertex, self.fs)
        elif self.procedure_state == ProcedureState.DEGENERATION_STATE:
            idxs_tbc = [v[1] for v in self.epsilon_neighbourhood if v[0]]
            self.seq = self.fvs.specify_vectors_on_exit_edge(self.l, self.hold_direction, self.bases, idxs_tbc, True)
            idxs_tbc = [v[1] for v in self.epsilon_neighbourhood if not v[0]]
            self.seq += self.uvs.specify_vectors_on_exit_edge(self.l, self.hold_direction, self.bases, self.vertex, self.fs, idxs_tbc, True)
        elif self.procedure_state == ProcedureState.END_DEGENERATION_STATE:
            idxs_tbc = set(range(len(self.fvs.vectors))) - set([v[1] for v in self.epsilon_neighbourhood if v[0]])
            self.seq = self.fvs.specify_vectors_on_exit_edge(self.l, self.hold_direction, self.bases, idxs_tbc)
            idxs_tbc = set(self.fs.features) - set([v[1] for v in self.epsilon_neighbourhood if not v[0]])
            self.seq += self.uvs.specify_vectors_on_exit_edge(self.l, self.hold_direction, self.bases, self.vertex, self.fs, idxs_tbc)

        if self.lv[0]:
            self.fvs.products_fv_B1l[self.lv[1]] = self.bases.dot_B1_fv(self.l, self.lv[1])
            if not self.hold_direction:
                self.fvs.products_fv_B1l[self.lv[1]] *= -1


    def _find_new_base_vector(self):
        self._find_sequence_on_exit_edge()
        if len(self.seq) == 0:
            if self.procedure_state != ProcedureState.DEGENERATION_STATE:
                return False
            else:
                return self._end_degeneration()
        else:
            gradient_zeroed = self._go_to_new_vertex()
            if not gradient_zeroed and self.procedure_state == ProcedureState.DEGENERATION_STATE:
                return self._end_degeneration()
        return True


    def _find_exit_edge(self):
        self._update_products_B1_cv()

        cv_projections = self.products_B1_cv[self.fs.features]
        fv_selection = self.bases.B_type[self.fs.features] & (cv_projections < 0)
        cv_projections[fv_selection] *= -1 
        cv_projections[fv_selection] -= self.fvs.sample_weight[self.bases.B_index[self.fs.features[fv_selection]]]

        ev_selection = (self.bases.B_type[self.fs.features] == False) & (cv_projections < 0)
        cv_projections[ev_selection] *= -1
        if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
            cv_projections[ev_selection] -= 2 * self.p_lambda * self.fs.feature_weight[self.bases.B_index[self.fs.features[ev_selection]]]

        argmax_cv_projections = np.argmax(cv_projections)
        if cv_projections[argmax_cv_projections] <= ZERO:
            return False
        self.l = self.fs.features[argmax_cv_projections]
        return self._check_stop_and_init_s()


    def _update_to_new_vertex(self):
        B1_l = self.bases.get_B1_vector(self.l)

        # base vector exchange
        self.bases.change_base(self.l, self.lv, self.kv)

        if self.procedure_state in (ProcedureState.DEGENERATION_STATE, ProcedureState.BEGIN_DEGENERATION_STATE):
            # remove the vector kv entering the base from the degeneration sequence
            self.epsilon_neighbourhood.remove((self.kv[0], self.kv[1]))
            # add the vector lv outgoing from the base from the degeneration sequence
            if self.procedure_state == ProcedureState.DEGENERATION_STATE:
                self.epsilon_neighbourhood.add((self.lv[0], self.lv[1]))

        # coordinates of the new vertex
        indexes_of_zero_values = self.fs.features[np.where(self.uvs.in_base[self.fs.features])[0]]
        indexes_of_real_values = self.fs.features[np.where(~self.uvs.in_base[self.fs.features])[0]]
        self.vertex[indexes_of_zero_values] = 0.
        correction = self.kv[2] * B1_l[indexes_of_real_values]
        if self.hold_direction:
            self.vertex[indexes_of_real_values] += correction
        else:
            self.vertex[indexes_of_real_values] -= correction

        # update fvs.products_fv_vertex
        if self.procedure_state == ProcedureState.NORMAL_STATE:
            self.fvs.update_products_fv_vertex(self.kv)
        else:
            idxs_tbc = [v[1] for v in self.epsilon_neighbourhood if v[0]]
            self.fvs.update_products_fv_vertex(self.kv, idxs_tbc)
            if self.lv[0]:
                self.fvs.recalculate_product_fv_vertex(self.lv[1], self.vertex, self.fs)

        if self.procedure_state == ProcedureState.BEGIN_DEGENERATION_STATE:
            self.procedure_state = ProcedureState.DEGENERATION_STATE


    def _find_epsilon_neighbourhood(self):
        """help with checking degeneration"""

        kv_seq_id = self.kv[3]
        self.epsilon_neighbourhood = set()

        if (kv_seq_id > 0) and EQUALS_EPSILON(self.kv[2], self.seq[kv_seq_id-1][2]):
            i = kv_seq_id-1
            while (i >= 0) and EQUALS_EPSILON(self.kv[2], self.seq[i][2]):
                self.epsilon_neighbourhood.add(self.seq[i][:2])
                i -= 1

        if (kv_seq_id < len(self.seq)-1) and (EQUALS_EPSILON(self.kv[2], self.seq[kv_seq_id+1][2])):
            new_kv = self.seq[kv_seq_id]
            i = kv_seq_id+1
            while (i < len(self.seq)) and EQUALS_EPSILON(self.kv[2], self.seq[i][2]):
                self.epsilon_neighbourhood.add(new_kv[:2])
                if new_kv[0]:
                    if self.fvs.products_fv_B1l[new_kv[1]] <= 0:
                        self._modify_cv_minus(new_kv[1])
                else:
                    self._modify_cv_unit(new_kv[1])
                new_kv = self.seq[i]
                if new_kv[0]:
                    if self.fvs.products_fv_B1l[new_kv[1]] > 0:
                        self._modify_cv_plus(new_kv[1])
                    self.s -= self._get_cv_projection_correction_by_fv(new_kv[1])
                else:
                    self.s -= self._get_cv_projection_correction_by_uv(new_kv[1])
                    self._modify_cv_unit(new_kv[1])
                i += 1
            self.kv = new_kv

        return len(self.epsilon_neighbourhood)


    def _check_degeneration(self):
        """check and eventually initiate the degeneration state"""

        # try to create epsilon neighbourhood sequence
        if self._find_epsilon_neighbourhood() == 0:
            return False

        self.procedure_state = ProcedureState.BEGIN_DEGENERATION_STATE

        # move behind the last vertex
        self.epsilon_neighbourhood.add((self.kv[0], self.kv[1]))
        if self.kv[0]:
            if self.fvs.products_fv_B1l[self.kv[1]] <= 0:
                self._modify_cv_minus(self.kv[1])
        else:
            self._modify_cv_unit(self.kv[1])

        # spreading the edges and calculating the distances after spreading
        self.seq = []
        for v in self.epsilon_neighbourhood:
            if v[0]:
                vdist = (2 + v[1] - self.fvs.products_fv_vertex[v[1]]) / self.fvs.products_fv_B1l[v[1]]
            else:
                vdist = (len(self.fs.features) + 1 + v[1] - self.vertex[v[1]]) / self.bases.get_B1_value(v[1], self.l)
            self.seq.append((v[0], v[1], vdist))

        # sort sequence on exit edge
        self.seq.sort(key=lambda x: x[2])

        # moving to the optimal temporary vertex in degeneracy
        for i, v in enumerate(self.seq):
            if v[0]:
                self.s += self._get_cv_projection_correction_by_fv(v[1])
                if self.fvs.products_fv_B1l[v[1]] < 0:
                    self._modify_cv_plus(v[1])
                elif (self.s < 0) and (i+1 < len(self.seq)):
                    self._modify_cv_minus(v[1])
            else:
                self.s -= self._get_cv_projection_correction_by_uv(v[1])
                self._modify_cv_unit(v[1])
            if self.s > 0:
                break

        # vector entering the base
        self.kv = self.seq[i][0], self.seq[i][1], self.seq[i][2], i

        return True


    def _end_degeneration(self):
        """ending of degeneration and return to normal state"""

        # modify the correction vector after passing through the last vector
        if len(self.seq) > 0:
            if self.kv[0]:
                if self.fvs.products_fv_B1l[self.kv[1]] <= 0:
                    self._modify_cv_minus(self.kv[1])
            else:
                self._modify_cv_unit(self.kv[1])

        # determination of the non-moving vertex and the products of <fv,v>
        self._optimal_vertex()

        # finding a new vertex
        self.procedure_state = ProcedureState.END_DEGENERATION_STATE
        res = self._find_new_base_vector()
        self.epsilon_neighbourhood = set()
        self.procedure_state = ProcedureState.NORMAL_STATE

        return res


    def _optimal_vertex(self):
        """determination of the base vertex not moved and products of <fv,v> in the non-moving vertex"""
        self.vertex = self.bases.get_current_vertex()
        self.fvs.recalculate_products_fv_vertex(self.vertex, self.fs)


    def _optimize(self):
        if self.s <= 0:
            return
        while True:
            if not self._find_new_base_vector():
                break
            if self.procedure_state == ProcedureState.NORMAL_STATE:
                self._check_degeneration()
            self._update_to_new_vertex()
            if self.__verbose == 1:
                self._diagnose()
            if not self._find_exit_edge():
                break

        if self.procedure_state == ProcedureState.DEGENERATION_STATE:
            self._optimal_vertex()
            self.procedure_state = ProcedureState.NORMAL_STATE


    def _func_crit(self, vertex):
        pyvs = np.dot(self.fvs.vectors[:,self.fs.features], vertex[self.fs.features])
        cr1 = sum([sw * (1. - pyv) for pyv, sw in zip(pyvs, self.fvs.sample_weight) if (pyv+1e-10 < 1)])
        cr2 = self.p_lambda * sum([abs(vertex[f])*self.fs.feature_weight[f]
                                  for f in self.fs.features if self.uvs.in_base[f] == False])
        return cr1, cr2, cr1 + cr2


    def _margin_width(self, vertex):
        return 1 / sum([abs(vertex[f])*self.fs.feature_weight[f] for f in self.fs.features if self.uvs.in_base[f] == False])
    

    def _correction_vector(self):
        if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
            cv = -self.p_lambda * self.uvs.ev[self.fs.features] * self.fs.feature_weight[self.fs.features]
        else:
            cv = [0 for f in self.fs.features]
        cv += (self.fvs.vectors[self.fvs.on_positive_side][:,self.fs.features] * np.array([self.fvs.sample_weight[self.fvs.on_positive_side]]).T).sum(axis=0)
        return cv


    def _diagnose(self):
        self.step += 1
        print("step: ", self.step)
        vertex = self.bases.get_current_vertex()
        # important_features = [f for f in self.fs.features if not self.uvs.in_base[f]]
        #print("--------------")
        print("Fc = ", self._func_crit(vertex))
        print("Margin = ", self._margin_width(vertex))
        # print("procedure state = ", self.procedure_state)
        # if self.procedure_state == ProcedureState.DEGENERATION_STATE:
        #     print()
        #     print(self.epsilon_neighbourhood)
        # print()
        # print("l={}, lv={}, kv={}".format(self.l, self.lv, self.kv))
        # print()
        # print("seq: ", self.seq)
        # print()
        # self.bases.diagnose()
        # print()
        # print("cv(proc): ", self.cv[self.fs.features])
        # print("cv(spr_): ", self._correction_vector())
        # for cv_p, cv_s in zip(self.cv[self.fs.features],self._correction_vector()):
        #     if not EQUALS_EPSILON(cv_p, cv_s):
        #         print("--->", cv_p, cv_s) 
        # print()
        # print("B1 x cv: ", self.products_B1_cv[self.fs.features])
        # print()
        # print("vertex(spr_): ", [(f,v) for f,v in zip(self.fs.features, vertex[self.fs.features]) if f in important_features])
        # print("vertex(proc): ", [(f,v) for f,v in zip(self.fs.features, self.vertex[self.fs.features]) if f in important_features])
        # print("feature weight: ", self.fs.feature_weight[self.fs.features])
        # print()
        # print("# proc_ps man_ps man_pyv error")
        # for i,(fv,ps) in enumerate(zip(self.fvs.vectors, self.fvs.on_positive_side)):
        #     pyv = np.dot(fv[self.fs.features],vertex[self.fs.features])
        #     print(i, ps, pyv+1e-10 < 1, pyv, "<---" if ps != (pyv+1e-10 < 1) else "")
        # print("--------------")



class SekwemClassifier(CPLBaseClassifier):

    def __repr__(self) -> str:
        return f"SekwemClassifier(C={self._C})"


    def _method_fit(self):
        self.fs.set_full()
        self.bases.init_full()

        # correction_vector
        self.cv = (self.fvs.vectors * np.array([self.fvs.sample_weight]).T).sum(axis=0)
        if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
            self.cv += -self.p_lambda * self.uvs.ev * self.fs.feature_weight

        # <B1,correction_vector> products
        self.products_B1_cv = np.copy(self.cv)

        # determination of the first base direction
        self.l = abs(self.cv).argmax()
        self._check_stop_and_init_s()

        # search for the optimal vertex
        self._optimize()



class GenetClassifier(CPLBaseClassifier):

    def __repr__(self) -> str:
        return f"GenetClassifier(C={self._C})"


    def _method_fit(self):
        self.bases.init_empty()

        # correction_vector
        self.cv = (self.fvs.vectors * np.array([self.fvs.sample_weight]).T).sum(axis=0)
        if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
            self.cv += -self.p_lambda * self.uvs.ev * self.fs.feature_weight
        

        # determination of the first feature
        m = abs(self.cv).argmax()

        # optimization
        while True:
            if not self.__add_feature(m):
                break
            self._optimize()
            m = self.__find_feature_to_extend()
            if m is None:
                break


    def __add_feature(self, m) -> bool:
        """Add feature to feature space and prepare to optimization
        m - index of the feature to be added
        returns - whether the procedure can be optimised after the addition of the feature
        """
        self.fs.add_feature(m)
        self.l = m

        # setting parameters in the new space

        # vertex v
        self.vertex[m] = 0.0

        # correction vector
        self.cv[m] = sum(self.fvs.vectors[self.fvs.on_positive_side,m] * self.fvs.sample_weight[self.fvs.on_positive_side])
        if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
            self.cv[m] += -self.p_lambda * self.uvs.ev[m] * self.fs.feature_weight[m]
        
        # bases
        self.bases.extend_by_feature(m)
        self.products_B1_cv[m] = self.bases.dot_B1_realv(m, self.cv)

        return self._check_stop_and_init_s()


    def __find_feature_to_extend(self) -> int:
        """Find a feature to extend the feature space
        returns - index of the feature or None
        """
        fos = self.fs.features_outside_space
        wc = (self.fvs.vectors[self.fvs.on_positive_side][:,fos] * np.array([self.fvs.sample_weight[self.fvs.on_positive_side]]).T).sum(axis=0)

        ffvb = self.fs.features[self.bases.B_type[self.fs.features]]
        wc -= (self.products_B1_cv[ffvb] * (self.fvs.vectors[self.bases.B_index[ffvb]][:,fos]).T).sum(axis=1)

        if self._optimization_mode == OptimizationMode.OPT_HYP_MODE:
            wc -= self.p_lambda * self.uvs.ev[fos] * self.fs.feature_weight[fos]
        
        correction_mask = (wc < 0)
        wc[correction_mask] = -wc[correction_mask] - 2 * self.p_lambda * self.fs.feature_weight[fos[correction_mask]]

        if (len(wc) == 0) or (np.max(wc) < ZERO):
            return None
        else:
            return fos[wc.argmax()]

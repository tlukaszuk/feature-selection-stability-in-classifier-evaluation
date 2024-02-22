import numpy as np

from .consts import INFINITY, ZERO, EPSILON, EQUAL_ZERO


class Bases:
    """Information on the status of the primary base and inverse base
    
    Attributes
    ----------
    B_type : np.array of shape (dim)
        Type of vector in primary base: True - feature vector, False - unit vector
    B_index : np.array of shape (dim)
        Index of vector in primary base - index of feature or unit vector
    B1 : np.array of shape (dim,max_fv)
        Inverse base, its columns
    """
    
    def __init__(self, num, dim, fvs, uvs, fs):
        self.__num = num
        self.__dim = dim
        self.__fvs = fvs
        self.__uvs = uvs
        self.__fs = fs
        self.__max_fv = min(self.__num, self.__dim) + 1

        self.B_type = np.full((self.__dim), False)
        self.B_index = np.arange(self.__dim)
        self.B1 = np.zeros((self.__dim, self.__max_fv))
        
        # __fvs - feature vectors
        # __uvs - unit vectors
        # __fs - feature space
        # __max_fv - maximum number of real rows in B1
        # __uv_info - information on the presence of uvs in B
        #           uv_info[i]==j means unit vector i is in j row in B
        #           uv_info[i]==-1 means unit vector i is outside B
        # __uv_ev - information on the direction of unit vectors in bases B and B1
        
        
    def init_empty(self):
        # base
        self.__uvs.in_base[:] = False
        
        # inverse base
        self.__actual_fv = np.zeros(0, dtype=int)  #rows with real values in the B1 base  
        self.__len_actual_fv = 0
        self.__uv_info = np.full((self.__dim), -1)
        self.__uv_ev = np.full((self.__dim), 0.0)
        
        
    def init_full(self):
        # base
        self.__uvs.in_base[:] = True
        
        # inverse base
        self.__actual_fv = np.zeros(0, dtype=int)
        self.__len_actual_fv = 0
        self.__uv_info = np.arange(self.__dim)
        self.__uv_ev = np.full((self.__dim), 1.0)
        
        
    def extend_by_feature(self, m):
        self.B_type[m], self.B_index[m] = False, m
        self.__uvs.in_base[m] = True
        self.__uvs.ev[m] = True
        
        self.__uv_info[m] = m
        self.__uv_ev[m] = 1.
        
        features_fv = [f for f in self.__fs.features[:-1] if self.B_type[f]]
        for i in range(self.__len_actual_fv):
            self.B1[m,i] = -sum([self.B1[f,i] * self.__fvs.vectors[self.B_index[f],m] for f in features_fv])
     
    
    def get_B1_vector(self, col):
        """Return the vector of base B1 as a full size vector with real numbers
        """
        v = np.zeros(self.__dim)
        v[self.__actual_fv] = self.B1[col,:self.__len_actual_fv]
        uv_id = self.__uv_info[col]
        if uv_id != -1:
            v[uv_id] = self.__uv_ev[uv_id]
        return v
    
    
    def get_B1_value(self, row, col):
        if not row in self.__actual_fv:
            uv_id = self.__uv_info[col]
            if uv_id == row:
                return self.__uv_ev[uv_id]
            else:
                return 0.0
        else:
            return self.B1[col,np.where(self.__actual_fv == row)[0][0]]
    
    
    def dot_B1_realv(self, col, realv):
        """Calculate the product of the vectors B1[col] and single real vector
        """
        dot = np.dot(self.B1[col,:self.__len_actual_fv], realv[self.__actual_fv])
        uv_id = self.__uv_info[col]
        if uv_id != -1:
            dot += self.__uv_ev[uv_id] * realv[uv_id]
        return dot


    def dots_B1s_realv(self, realv):
        """Calculate the products of the all vectors B1 and single real vector
        """
        B1_cols = self.B1[self.__fs.features,:self.__len_actual_fv]
        dots = np.dot(B1_cols, realv[self.__actual_fv])
        idx_change_needed = self.__fs.features[self.__uv_info[self.__fs.features] != -1]
        if len(idx_change_needed) > 0:
            uv_ids = self.__uv_info[idx_change_needed]
            dots[self.__fs.features_positions(idx_change_needed)] += self.__uv_ev[uv_ids] * realv[uv_ids]
        return dots
    
    
    def dot_B1_fv(self, col, fv_id):
        return self.dot_B1_realv(col, self.__fvs.vectors[fv_id])


    def dots_B1s_fv(self, fv_id):
        return self.dots_B1s_realv(self.__fvs.vectors[fv_id])
    
    
    def dot_B1_uv(self, col, uv_id):
        return (self.__uvs.ev[uv_id] if not self.__uvs.in_base[uv_id] else self.__uv_ev[uv_id]) * self.get_B1_value(uv_id, col)


    def dots_B1s_uv(self, uv_id):
        if not uv_id in self.__actual_fv:
            B1_row = np.zeros(len(self.__fs.features))
            B1_row[self.__uv_info.index(uv_id)] = self.__uv_ev[uv_id]
        else:
            fv_index = np.where(self.__actual_fv == uv_id)[0][0]
            B1_row = self.B1[self.__fs.features,fv_index]
        return (self.__uvs.ev[uv_id] if not self.__uvs.in_base[uv_id] else self.__uv_ev[uv_id]) * B1_row
        
        
    def dots_fvs_B1(self, fv_ids, col):
        """Calculate the products of the multi feature vectors indicated by fv_ids and the vectors B1[col]
        """
        fvs = self.__fvs.vectors[fv_ids]
        dots = np.dot(fvs[:,self.__actual_fv], self.B1[col,:self.__len_actual_fv])
        uv_id = self.__uv_info[col]
        if uv_id != -1:
            dots += self.__uv_ev[uv_id] * fvs[:,uv_id]
        return dots


    def get_current_vertex(self):
        """Determine, without iterative calculations, the coordinates of the current vertex
           where the opimisation procedure is located
        """
        vertex = np.zeros(self.__dim)
        vertex[self.__actual_fv] = np.array([r[:self.__len_actual_fv] for r,bv in zip(self.B1, self.B_type) if bv]).sum(axis=0)
        return vertex
    
    
    def change_base(self, l, lv, kv):
        # 1. update B1
        # 1.1
        if lv[0] == False:
            idx = len(self.__actual_fv)
            self.B1[self.__fs.features,idx] = 0
            self.B1[l,idx] = self.__uv_ev[lv[1]]
            self.__uv_info[l] = -1
            self.__actual_fv = np.append(self.__actual_fv, [lv[1]])
            self.__len_actual_fv += 1
        
        dots_B1s_kv = (self.dots_B1s_uv, self.dots_B1s_fv)[kv[0]](kv[1])
        l_position = self.__fs.feature_position(l)
        rows_B1_wl = np.delete(self.__fs.features, l_position)

        # 1.2 modification of column B1[l]
        b = 1.0 / dots_B1s_kv[l_position]
        self.B1[l,:self.__len_actual_fv] *= b
        
        # 1.3 modification of the other columns B1
        self.B1[rows_B1_wl,:self.__len_actual_fv] -= np.array([np.delete(dots_B1s_kv, l_position)]).T * self.B1[l,:self.__len_actual_fv]
        
        # 1.4
        if kv[0] == False:
            b_begin = np.where(self.__actual_fv == kv[1])[0][0]
            b_end = len(self.__actual_fv)-1
            self.B1[self.__fs.features,b_begin:b_end] = self.B1[self.__fs.features,b_begin+1:b_end+1]
            self.__actual_fv = np.delete(self.__actual_fv, b_begin)
            self.__len_actual_fv -= 1
            self.__uv_info[l] = kv[1]
            self.__uv_ev[kv[1]] = self.__uvs.ev[kv[1]]
        
        # 2. update B
        if lv[0]:
            self.__fvs.in_base[lv[1]] = False
        else:
            self.__uvs.in_base[lv[1]] = False
        self.B_type[l], self.B_index[l] = kv[:2]
        if kv[0]:
            self.__fvs.in_base[kv[1]] = True
        else:
            self.__uvs.in_base[kv[1]] = True
            
    
    def diagnose(self):
        print("B:")
        for f in self.__fs.features:
            if self.B_type[f] or (self.B_index[f] != f):
                print(f, self.B_type[f], self.B_index[f])
        # print()
        # print("B1 real rows:")
        # for i,row in enumerate(self.__actual_fv):
        #     print(row, [round(self.B1[f][i],2) for f in self.__fs.features][:20], "..." if len(self.__fs.features)>20 else "")
        #     #print(row, [round(self.B1[f][i],2) for f in self.__fs.features])
        # print()
        # print("Checking BxB1:")
        # for i,f_B in enumerate(self.__fs.features):
        #     v_B_type, v_B_index = self.B_type[f_B], self.B_index[f_B] 
        #     row = [(self.dot_B1_uv, self.dot_B1_fv)[v_B_type](f, v_B_index) for f in self.__fs.features]
        #     row_ok = (EQUAL_ZERO(row[i]-1.) and (EQUAL_ZERO(sum(row)-1.)))
        #     if not row_ok:
        #         print("error in row {} [{}] {}".format(i, sum(row), [round(ri,2) for ri in row]))
            

    
    
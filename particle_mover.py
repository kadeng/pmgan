import math

import numpy as np
import ot  # Python Optimal Transport Library ( conda install -c conda-forge pot )
import torch  # pytorch, see pytorch.org
import torch.nn as nn
import torch.optim.sgd
from scipy.misc import logsumexp as np_logsumexp
from torch.autograd import Variable, Function
from pmgan.pm_utils import lgamma, _float32_eps, _pi, _pi_var, L1Distance, L2Distance
from typing import List, Dict, Tuple, Any, Callable
from collections import defaultdict
import torch
import random


class ParticleMoverDistances(object):
    # @TODO: Look into Ball Trees or Vantage Point Trees ( http://stevehanov.ca/blog/index.php?id=130 ) to improve runtime complexity
    samples_x : torch.FloatTensor
    samples_y : torch.FloatTensor
    distances_xy : List[List[Variable]]
    distances_yx : List[List[Variable]]
    distances_xx : List[List[Variable]]
    distances_yy : List[List[Variable]]
    nnxx : List[np.ndarray]
    nnxy : List[np.ndarray]
    nnyx : List[np.ndarray]
    nnyy : List[np.ndarray]
    marriages_xy : List[int]
    marriages_yx : List[int]
    ot_solution : List[Tuple[float,float,int,int]]
    log_density_precalc_xy : List[Tuple[Variable, Variable, List[int]]]
    log_density_precalc_xx : List[Tuple[Variable, Variable, List[int]]]
    log_density_precalc_yx : List[Tuple[Variable, Variable, List[int]]]
    log_density_precalc_yy : List[Tuple[Variable, Variable, List[int]]]
    log_log_density_xx : List[float]
    log_log_density_yy : List[float]
    log_log_density_xy : List[float]
    log_log_density_yx : List[float]

    def __init__(self):
        super(ParticleMoverDistances, self).__init__()


    @staticmethod
    def _precompute_distances(samples_x : torch.FloatTensor, samples_y : torch.FloatTensor, dist_fn : Callable) -> List[List[torch.FloatTensor]]:
        distances = []
        nan = float('nan')
        for ai in range(samples_x.size()[0]):
            xy = []
            a = samples_x[ai]
            for bi in range(samples_y.size()[0]):
                b = samples_y[bi]
                distance = dist_fn(a, b)
                xy.append(distance)
            distances.append(xy)
        return distances

    def set_metric(self, distance : nn.Module, w : Variable):
        self.distance = distance
        self.w = w
        self.log_volume_constant = w * 0.5 * _pi_var - lgamma(1.0 + w / 2.0)

    def set_samples(self, samples_x : torch.FloatTensor, samples_y : torch.FloatTensor, clear_marriages=True):
        '''
        Set sample lists and precalculate all particle distances
        '''

        self.samples_x = samples_x
        self.samples_y = samples_y
        self.distances_xy = self._precompute_distances(samples_x, samples_y, self.distance)
        self.distances_yx = self._list_matrix_transpose(self.distances_xy)
        self.distances_xx = self._precompute_distances(samples_x, samples_x, self.distance)
        self.distances_yy = self._precompute_distances(samples_y, samples_y, self.distance)

        self.nnxy = self._distance_matrix_argsort(self.distances_xy)
        self.nnyx = self._distance_matrix_argsort(self.distances_yx)
        self.nnxx = self._distance_matrix_argsort(self.distances_xx)
        self.nnyy = self._distance_matrix_argsort(self.distances_yy)

        # Reset state derived from samples
        if clear_marriages:
            self.marriages_xy = None
            self.marriages_yx = None
        self.ot_solution = None
        self.dmat = None
        self.px = None
        self.py = None
        self.log_density_precalc_xx = None
        self.log_density_precalc_yx = None
        self.log_density_precalc_yy = None
        self.log_density_precalc_xy = None
        self.log_density_xx = None
        self.log_density_xy = None
        self.log_density_yx = None
        self.log_density_yy = None
        self.buddies = None

    def precalc_densities(self, max_kx : int=None, max_ky : int=None):
        samples_x = self.samples_x
        samples_y = self.samples_y
        if max_kx is None:
            max_kx = 5 + len(samples_x)//50
        if max_ky is None:
            max_ky = 5 + len(samples_y)//50
        self.log_density_precalc_xx = [ self.nn_log_density_estimate( i, self.distances_xx, self.nnxx, max_k=max_kx) for i in range(len(samples_x))] # Densities of x's in P_x
        self.log_density_precalc_yx = [ self.nn_log_density_estimate( i, self.distances_yx, self.nnyx, max_k=max_kx) for i in range(len(samples_y))] # Densities of y's in P_x
        self.log_density_precalc_yy = [ self.nn_log_density_estimate( i, self.distances_yy, self.nnyy, max_k=max_ky, sample_from_y=True) for i in range(len(samples_y))] # Densities of y's in P_y
        self.log_density_precalc_xy = [ self.nn_log_density_estimate( i, self.distances_xy, self.nnxy, max_k=max_ky) for i in range(len(samples_x))] # Densities of x's in P_y

        self.log_density_xx = [ v[0].data[0] for v in self.log_density_precalc_xx] #!NORMALIZE !!!!!
        self.log_density_yx = [ v[0].data[0] for v in self.log_density_precalc_yx]
        self.log_density_xy = [ v[0].data[0] for v in self.log_density_precalc_xy]
        self.log_density_yy = [ v[0].data[0] for v in self.log_density_precalc_yy]

    def solve_optimal_transport(self) -> bool:
        if self.log_density_yy is None:
            self.precalc_densities()

        # We want double precision for leaving log-space exponentiation
        vx = np.array(self.log_density_xx + self.log_density_yx ).astype(np.double)
        vy = np.array(self.log_density_xy + self.log_density_yy ).astype(np.double)
        vx -= np_logsumexp(vx)
        vy -= np_logsumexp(vy)
        px = np.exp(vx) # Densities of samples of x and y in P_x
        py = np.exp(vy) # Densities of samples of x and y in P_y

        # Matrix of distances between samples. Built from 4 distance matrices in it's quadrants.
        xr = torch.cat([self._to_tensor_mat(self.distances_xx), self._to_tensor_mat(self.distances_xy)], dim=1)
        yr = torch.cat([self._to_tensor_mat(self.distances_yx), self._to_tensor_mat(self.distances_yy)], dim=1)
        self.px = px
        self.py = py
        self.dmat = torch.cat([xr,yr], dim=0).numpy() # Distance of samples.
        solution = ot.emd(px, py, self.dmat) # Solution of transport problem
        snz = np.nonzero(solution) # Take nonzero values only
        if len(snz[0])==0:
            self.ot_solution = None
            return False
        else:
            # Create & remember a list of tuples of ( transport_mass, distance, index_i, index_j ) sorted in descending order
            self.ot_solution = sorted(list(zip(solution[snz[0], snz[1]].tolist(), self.dmat[snz[0], snz[1]].tolist(), snz[0].tolist(), snz[1].tolist())), reverse=True)
            return True

    def set_sources(self, sources, source_distance_fn):
        self.sources = sources
        self.source_distance_fn = source_distance_fn
        self.source_distances = self._precompute_distances(sources, sources, source_distance_fn)
        self.source_nn = self._distance_matrix_argsort(self.source_distances)
        self.buddies = None
        self.cluster_targets = None


    def calculate_weighted_cluster_targets_yx(self, weights=[0.35, 0.2, 0.15, 0.1, 0.1, 0.1 ]):
        self.cluster_targets = torch.zeros_like(self.sources.data)
        if self.marriages_yx is not None:
            for j,i in self.marriages_yx:
                for k, w in enumerate(weights):
                    self.cluster_targets[j] += self.sources.data[self.source_nn[i][k]] * w
        self.cluster_distances_yx = self._precompute_distances(Variable(self.cluster_targets, requires_grad=False), self.sources, self.source_distance_fn)
        self.cluster_nn_yx = self._distance_matrix_argsort(self.cluster_distances_yx)


    def refine_marriages_yx(self,  weights=[0.35, 0.2, 0.15, 0.1, 0.1, 0.1 ], method='stable' ):
        self.calculate_weighted_cluster_targets_yx(weights=weights)
        if method=='stable':
            new_marriages = self._gale_shapley_stable_marriages(self.cluster_distances_yx, self.cluster_nn_yx, len(self.samples_x))
        elif method=='unlikely':
            new_marriages = self._unlikely_marriages_algorithm(self.cluster_distances_yx, self.cluster_nn_yx, len(self.samples_x))
        else:
            raise Exception("Unknown marriage refinement method %s" % (method))
        self.prev_marriages_yx = self.marriages_yx
        self.marriages_yx = new_marriages

    def refine_marriages_by_source(self, method='stable', refinement='switch', max_distance_change=1.000):
        d = self.source_distances
        nn = self.source_nn
        n = len(nn)
        if method=='stable':
            buddies = self._gale_shapley_stable_marriages(d, nn, n, allow_same=False)
        elif method=='unlikely':
            buddies = self._unlikely_marriages_algorithm(d, n, allow_same=False, ignore_equal=False)
        else:
            raise Exception("Unknown marriage refinement method %s" % (method))
        if self.marriages_xy is not None:
            marriages = dict(self.marriages_xy)
            swaps_xy = self._find_buddy_swaps(self.distances_xy, marriages, max_distance_change, buddies)
            for i, k in enumerate(self.marriages_xy):
                if k in swaps_xy:
                    self.marriages_xy[i]=swaps_xy[k][0]
            print("Swapped %d entries in xy marriages" % (len(swaps_xy)))

        if self.marriages_yx is not None:
            marriages2 = dict([(h[1],h[0]) for h in self.marriages_yx])
            swaps_yx = self._find_buddy_swaps(self.distances_xy, marriages2, max_distance_change, buddies)
            for i in range(len(self.marriages_yx)):
                sk = self.marriages_yx[i]
                k = (sk[1],sk[0]) # Swap indices i and j
                if k in swaps_yx:
                    st = swaps_yx[k][0]
                    self.marriages_yx[i]=(st[1],st[0]) # Swap indices i and j
            print("Swapped %d entries in yx marriages" % (len(swaps_yx)))


    def _find_buddy_swaps(self, d, marriages, max_distance_change, buddies, inv=False):
        swaps = {}
        for i1, i2 in buddies:
            j1 = marriages.get(i1, None)
            j2 = marriages.get(i2, None)
            if j1 is None or j2 is None:
                continue
            dist = d[i1][j1] + d[i2][j2]
            swap_dist = d[i1][j2] + d[i2][j1]
            swap_distance_ratio = swap_dist.data[0] / (dist.data[0] + _float32_eps)
            if swap_distance_ratio <= max_distance_change:
                swaps[(i1, j1)] = ((i1, j2), swap_distance_ratio)
                swaps[(i2, j2)] = ((i2, j1), swap_distance_ratio)


        return swaps

    def solve_marriages_xy(self, method='unlikely', friends=10):
        nn = self.nnxy
        nno = self.nnyx
        n = len(nn)
        d = self.distances_xy
        # Now collect distances of married pairs
        if method=='stable':
            self.marriages_xy = self._gale_shapley_stable_marriages(d, nn, n)
        elif method=='unlikely':
            self.marriages_xy = self._unlikely_marriages_algorithm(d, n, nn, nno, friends=friends)
        elif method=='social':
            self.marriages_xy = self._social_marriages_algorithm(d,n, nn, nno, friends=friends)
        else:
            raise Exception("Unknown marriage method %s" % (method))

    def solve_marriages_yx(self, method='unlikely', friends=0):
        nn = self.nnyx
        nno = self.nnxy
        n = len(nn)
        d = self.distances_yx
        # Now collect distances of married pairs
        if method=='stable':
            self.marriages_yx = self._gale_shapley_stable_marriages(d, nn, n)
        elif method=='unlikely':
            self.marriages_yx = self._unlikely_marriages_algorithm(d, n, nn, nno, friends=friends)
        elif method=='social':
            self.marriages_yx = self._social_marriages_algorithm(d,n, nn, nno, friends=friends)
        else:
            raise Exception("Unknown marriage method %s" % (method))


    def _radial_log_density_ratio_estimate_yx(self):
        if self.log_density_yx is None:
            self.precalc_densities()

        self.marriage_log_density_ratio_yx = []
        for i, j in self.marriages_yx:
            logp_y = self.log_density_yx[i]
            logp_x = self.log_density_xy[j]
            log_density_ratio = (logp_y - logp_x)
            self.marriage_log_density_ratio_yx.append(log_density_ratio)

    def _radial_log_density_ratio_estimate_xy(self):
        if self.log_density_xy is None:
            self.precalc_densities()

        self.marriage_log_density_ratio_xy = []
        for i, j in self.marriages_xy:
            logp_x = self.log_density_xy[i]
            logp_y = self.log_density_yx[j]
            log_density_ratio = (logp_x - logp_y)
            self.marriage_log_density_ratio_xy.append(log_density_ratio)

    # Better don't use this, unless the density estimation method is really appropriate
    def weighted_marriage_distances_xy(self, method='unlikely', density_ratio_estimate='radial', invert=False, max_weight=4.0, min_weight=0.5, friends=10):
        self.marriage_importances_xy = [1.0] * len(self.marriages_xy)
        if density_ratio_estimate=='radial':
            self._radial_log_density_ratio_estimate_xy()
        else:
            raise Exception("Unknown density ratio estimator %r" % (density_ratio_estimate))
        result = []
        d = self.distances_xy
        weights = []
        self.marriage_weights_xy = []
        wsum = 0.0
        for idx, pair in enumerate(self.marriages_xy):
            weight = max(min_weight, min(math.exp(self.marriage_log_density_ratio_xy[idx]), max_weight))
            weights.append(weight)
            wsum += weight
        wsum /= float(len(self.marriages_xy))
        for idx, pair in enumerate(self.marriages_xy):
            weight = weights[idx] / wsum
            i,j = pair
            if weight>0.0:
                result.append(d[i][j]*weight)
            self.marriage_weights_xy.append(weight)
        return torch.stack(result)

    # Better don't use this, unless the density estimation method is really appropriate
    def weighted_marriage_distances_yx(self, method='unlikely', density_ratio_estimate='radial', invert=False, max_weight=4.0, min_weight=0.5, friends=10):
        self.marriage_importances_yx = [1.0] * len(self.marriages_yx)
        if density_ratio_estimate=='radial':
            self._radial_log_density_ratio_estimate_yx()
        else:
            raise Exception("Unknown density ratio estimator %r" % (density_ratio_estimate))
        result = []
        d = self.distances_yx
        weights = []
        self.marriage_weights_yx = []
        wsum = 0.0
        for idx, pair in enumerate(self.marriages_yx):
            weight = max(min_weight, min(math.exp(self.marriage_log_density_ratio_yx[idx]), max_weight))
            weights.append(weight)
            wsum += weight
        wsum /= float(len(self.marriages_yx))
        for idx, pair in enumerate(self.marriages_yx):
            weight = weights[idx] / wsum
            i,j = pair
            if weight>0.0:
                result.append(d[i][j]*weight)
            self.marriage_weights_xy.append(weight)
        return torch.stack(result)


    def solve_buddies(self, method='stable'):
        d = self.source_distances
        nn = self.source_nn
        n = len(nn)
        if method=='stable':
            self.buddies = self._gale_shapley_stable_marriages(d, nn, n, allow_same=False)
        elif method=='unlikely':
            self.buddies = self._unlikely_marriages_algorithm(d, n, nn,nn, allow_same=False, ignore_equal=False)
        else:
            raise Exception("Unknown buddy finding method %s" % (method))

    def buddy_distances_xx(self, method='stable'):
        if self.buddies is None:
            self.solve_buddies(method=method)
        d = self.distances_xx
        result = []
        for i,j in self.buddies:
           result.append(d[i][j])
        return torch.stack(result)

    def buddy_distances_xy(self, method='stable'):
        if self.buddies is None:
            self.solve_buddies(method=method)
        d = self.distances_xy
        result = []
        mdict = dict(self.marriages_xy)
        used_i=set()
        used_j=set()
        for i1,i2 in self.buddies:
           j1 = mdict[i1]
           j2 = mdict[i2]
           if i1 in used_i:
               continue
           if i2 in used_i:
               continue
           assert(j1 not in used_j)
           assert(j2 not in used_j)
           used_i.add(i1)
           used_i.add(i2)
           used_j.add(j1)
           used_j.add(j2)
           result.append(d[i1][j2])
           result.append(d[i2][j1])
        return torch.stack(result)

    def buddy_distances_yx(self, method='stable'):
        if self.buddies is None:
            self.solve_buddies(method=method)
        d = self.distances_xy
        result = []
        mdict = dict([ (p[1],p[0]) for p in self.marriages_yx]) # invert order, so we get the right mapping direction
        used_i=set()
        used_j=set()
        for i1,i2 in self.buddies:
           j1 = mdict[i1]
           j2 = mdict[i2]
           if i1 in used_i:
               continue
           if i2 in used_i:
               continue
           assert(j1 not in used_j)
           assert(j2 not in used_j)
           used_i.add(i1)
           used_i.add(i2)
           used_j.add(j1)
           used_j.add(j2)
           result.append(d[i1][j2])
           result.append(d[i2][j1])
        return torch.stack(result)

    def marriage_distances_xy(self, method='unlikely'):
        result = []
        d = self.distances_xy
        for i,j in self.marriages_xy:
            result.append(d[i][j])
        return torch.stack(result)

    def marriage_distances_yx(self, method='unlikely'):
        result = []
        d = self.distances_yx
        for i,j in self.marriages_yx:
            result.append(d[i][j])
        return torch.stack(result)

    def nn_log_density_estimate(self, sample_idx, distance_matrix, nearest_neighbour_indices, ignore_same_index=True, k=1, max_k=6, radius_factor=1.5, min_r=32.0 * _float32_eps, sample_from_y=False):
        # Determine radius
        effective_k = k
        if ignore_same_index:
            effective_k += 1
            max_k += 1
        n = len(distance_matrix)
        effective_n = len(distance_matrix[sample_idx])
        if ignore_same_index:
            effective_n -= 1
        #knn_distance = distance_matrix[sample_idx][nearest_neighbour_indices[sample_idx][effective_k]]
        #max_knn_distance = distance_matrix[sample_idx][nearest_neighbour_indices[sample_idx][max_k]]

        # We pick a radius which is larger or equal than the knn distance to pick up more neighbours. But we never pick up more than max_k neighbours
        knn_indices = []
        last_dist = 0.0
        r = 0.0
        kc = 0

        for knn in range(0, n):
            if r>min_r and kc>=max_k:
                r = last_dist
                break
            knnidx = nearest_neighbour_indices[sample_idx][knn]
            if ignore_same_index and knnidx==sample_idx:
                continue
            dist = distance_matrix[sample_idx][knnidx].data[0]
            if dist>min_r and kc>=effective_k:
                r = (dist + last_dist) / 2.0
                break
            else:
                last_dist = dist
                r = dist
                kc += 1
                knn_indices.append(knnidx)

        if r<min_r:
            r = min_r
        r = Variable(torch.FloatTensor([float(r)]).type_as(self.log_volume_constant.data), requires_grad=False)
        log_k = Variable(torch.FloatTensor([math.log(float(len(knn_indices)))]).type_as(self.log_volume_constant.data))
        log_n = Variable(torch.FloatTensor([math.log(float(effective_n))]).type_as(self.log_volume_constant.data))
        log_density =  log_k - self.log_V_w(r) - log_n
        return log_density, r, knn_indices

    def log_V_w(self, r):
        '''
        Calculates logarithm of volume of w-dimensional hypersphere with radius r
        The dimensionality w has to be set by a call to set_metric(..)
        :param r: radius of hypersphere
        :return: logarithm of volume of w-dimensional hypersphere with radius r
        '''
        return self.log_volume_constant + torch.log(r) * self.w

    @staticmethod
    def _gale_shapley_stable_marriages(d, nn, n, allow_same=True):
        '''Internal implementation of Gale-Shapley Algorithm to find stable marriages.
           See https://en.wikipedia.org/wiki/Stable_marriage_problem
        '''
        remaining_men = set(range(n))
        proposal_count = [0]*n # Indexed by man's index
        fiancees = [None]*n # Indexed by woman's index
        while len(remaining_men)>0:
            for i in list(remaining_men):
                nni = nn[i]
                j = proposal_count[i]
                proposal_count[i]+=1
                idx = nni[j] # considered woman's index
                if not allow_same and i==idx:
                    continue
                fidx = fiancees[idx] # Is she engaged already ?
                if fidx is None: # No, then engage
                    fiancees[idx]=i
                    remaining_men.remove(i)
                else: # She is engaged, let's see whether she would like to change her mind.
                    current_distance = float(d[fidx][idx].data[0])
                    proposed_distance = float(d[i][idx].data[0])
                    # Is this a better match ?
                    if proposed_distance<current_distance:
                        remaining_men.add(fidx) # Put the disengaged man back on the marriage market
                        remaining_men.remove(i) # And remove the newly engaged man from the pool
                        fiancees[idx]=i # Mark them as engaged
        marriages = [-1]*n
        # We want to return them in the same indexing order as used by d and nn, so we invert the resulting indexing scheme
        for j, i in enumerate(fiancees):
            marriages[i]=j
        return list(enumerate(marriages))

    @staticmethod
    def _unlikely_marriages_algorithm(d, n, nn, nno, accuracy=0.001, allow_same=True, ignore_equal=True, friends=0):
        dtuples = []
        best_distance = defaultdict(lambda: 99999999.0)
        for i in range(n):
            for j in range(n):
                if not allow_same and i==j:
                    continue
                dtuples.append([99999999.0, float(d[i][j].data[0]), i, j])

        for db, dst, i, j in dtuples:
            best_distance[i] = min(dst, best_distance[i])
        for dt in dtuples:
            dt[0] = -best_distance[dt[2]] # we put negative best distance, so we sort them to get the samples from x with worst "best distance" first.

        dtsorted = sorted(dtuples)

        used_i = set()
        used_j = set()
        marriages = [-1]*n
        last_i = -1
        big_distance_counter = 0
        friend_marriages = 0
        normal_marriages = 0
        for ri in range(len(dtsorted)):
            bd, dst, i,j = dtsorted[ri]
            if i==last_i or i in used_i or j in used_j:
                continue
            if ignore_equal and dst<=accuracy:
                # If the samples are practically identical, we do not try to push them around anymore. This would be likely to make things worse
                used_i.add(i)
                used_j.add(j)
                continue
            used_i.add(i)
            used_j.add(j)
            marriages[i]=j
            last_i = i
            normal_marriages+=1

            if friends<1:
                continue
            # Now get intersecting friends married..
            friends_i = list(nn[i][1:friends+1])
            friends_j = set(nno[j][1:friends+1])
            for fi in friends_i:
                if  fi in used_i:
                    continue
                for fj in nn[fi][1:friends]:
                    if  fj in used_j:
                        continue
                    if fj in friends_j:
                        used_i.add(fi)
                        used_j.add(fj)
                        marriages[fi]=fj
                        friend_marriages+=1

        result = [(i,j) for i, j in enumerate(marriages) if j>=0]
        print("Marriages: %d of potentially %d - Normal: %d - Friend: %d" % (len(result), n, normal_marriages, friend_marriages))
        return result


    @staticmethod
    def _social_marriages_algorithm(d, n, nn, nno, friends=10):
        dtuples = []
        for i in range(n):
            for j in range(n):
                dtuples.append([d[i][j].data[0], i, j])
        dtsorted = sorted(dtuples)

        used_i = set()
        used_j = set()
        marriages = [-1]*n
        friend_marriages = 0
        normal_marriages = 0
        for ri in range(len(dtsorted)):
            dst, i,j = dtsorted[ri]
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            marriages[i]=j
            normal_marriages+=1
            # Now get intersecting friends married..
            friends_i = list(nn[i][1:friends+1])
            friends_j = set(nno[j][1:friends+1])
            for fi in friends_i:
                if  fi in used_i:
                    continue
                for fj in nn[fi][1:friends]:
                    if  fj in used_j:
                        continue
                    if fj in friends_j:
                        used_i.add(fi)
                        used_j.add(fj)
                        marriages[fi]=fj
                        friend_marriages+=1

        print("Normal Marriages: %d - Friend Marriages: %d" % (normal_marriages, friend_marriages))
        return list(enumerate(marriages))


    @staticmethod
    def _to_tensor_mat(dmat):
        '''
        Helper function to convert a list of lists of scalar pytorch tensors to a single pytorch matrix tensor
        '''
        len1 = len(dmat)
        len2 = len(dmat[0])
        result = torch.FloatTensor(len1,len2)
        for i in range(len1):
            for j in range(len2):
                result[i,j] = dmat[i][j].data[0]
        return result

    @staticmethod
    def _distance_matrix_argsort(distance_mat):
        '''
        Helper function to perform an argsort over the entries of our distance matrix
        '''
        return [ np.argsort(np.array([d.data[0] for d in row])) for row in distance_mat ]

    @staticmethod
    def _list_matrix_transpose(mat):
        '''
        Helper function to perform a matrix transpose on a list of lists
        '''
        len1 = len(mat)
        len2 = len(mat[0])
        res = [ [ None ]*len1 for j in range(len2) ]
        for i in range(len1):
            for j in range(len2):
                res[j][i] = mat[i][j]
        return res

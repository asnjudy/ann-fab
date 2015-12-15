import numpy as np

import nearpy.filters


class MatrixFilter(nearpy.filters.VectorFilter):
    def __init__(self, filter):
        self._filter = filter.filter_vectors

    def expand_item(self, v):
        vectors = v[0]
        data = v[1]
        if len(v) == 3:
            dists = v[2]
        else:
            dists = None

        if np.isscalar(vectors):
            return [v,]

        v_nd = len(vectors.shape)

        if v_nd == 1:
            return [v,]
        else:
            expanded = []
            if dists is None:
                for i in xrange(vectors.shape[0]):
                    expanded.append((vectors[i,:], data[i]))
            else:
                for i in xrange(vectors.shape[0]):
                    expanded.append((vectors[i,:], data[i], dists[i]))
            return expanded


    def expand(self, input_list):

        expanded_list = []

        n = len(input_list)
        if n == 0:
            return input_list

        for i in xrange(n):
            expanded = self.expand_item(input_list[i])
            for e in expanded:
                expanded_list.append(e)

        return expanded_list


    def filter_vectors(self, input_list):
        """
        Returns subset of specified input list.
        """
        return self._filter(self.expand(input_list))

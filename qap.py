__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
import utils
from scipy.stats.stats import pearsonr
from scipy.stats import linregress
import math
import itertools
import statsmodels.api as sm
import random

#######################################################################
# QAP
#######################################################################
class QAP():

    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y
        self.n = 0 if X is None and Y is None else Y.shape[0]
        self.Xmod = None

    def init(self):
        self.correlation(self.X, self.Y)
        #self.pearson(self.X, self.Y)

    def qap(self, random=True):
        if random:
            self.randomshuffle(5000)
        else:
            self.shuffle()
        self.correlation(self.Xmod, self.Y)
        #self.pearson(self.Xmod, self.Y)

    def correlation(self, x, y):
        xflatten = np.delete(x, [i*(self.n+1)for i in range(self.n)])
        yflatten = np.delete(y, [i*(self.n+1)for i in range(self.n)])
        pc = pearsonr(xflatten, yflatten)
        utils.printf('Pearson Correlation: {}'.format(pc[0]))
        utils.printf('p-value: {}'.format(pc[1]))

    def randomshuffle(self, times):
        self.Xmod = self.X.copy()
        permutations = list(itertools.permutations(range(self.n),2))
        for t in range(times):
            tuple = random.randint(0,len(permutations)-1)
            i = permutations[tuple][0]
            j = permutations[tuple][1]
            self.swap_cols(self.Xmod, i, j)
            self.swap_rows(self.Xmod, i, j)
        utils.printf('Permuted X: \n{}'.format(self.Xmod))

    def shuffle(self):
        self.Xmod = self.X.copy()
        permutations = itertools.permutations(range(self.n),2)
        for tuple in permutations:
            i = tuple[0]
            j = tuple[1]
            self.swap_cols(self.Xmod, i, j)
            self.swap_rows(self.Xmod, i, j)
        utils.printf('Permuted X: \n{}'.format(self.Xmod))

    def swap_cols(self, arr, frm, to):
        arr[:,[frm, to]] = arr[:,[to, frm]]

    def swap_rows(self, arr, frm, to):
        arr[[frm, to],:] = arr[[to, frm],:]

    def pearson(self, x, y):
        xflatten = np.delete(x, [i*(self.n+1)for i in range(self.n)])
        yflatten = np.delete(y, [i*(self.n+1)for i in range(self.n)])
        p = np.corrcoef(xflatten,yflatten)
        utils.printf('Pearson\'s correlation:\n{}'.format(p))

    def ols(self, x, y):

        xflatten = x.flatten()
        yflatten = y.flatten()
        xflatten = sm.add_constant(xflatten)
        model = sm.OLS(yflatten,xflatten)
        results = model.fit()
        print results.params
        print results.tvalues
        print results.t_test([1, 0])
        print results.f_test(np.identity(2))





__author__ = 'espin'

#######################################################################
# Dependencies
#######################################################################
import sys
import collections
import numpy as np
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from libs import utils
from libs.profiling import Profiling
import time
import gc
from scipy import stats
from scipy.stats.mstats import zscore

#######################################################################
# MRQAP
#######################################################################
INTERCEPT = 'Intercept'

class MRQAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, X=None, npermutations=-1, diagonal=False, directed=False, logfile=None, memory=None, standarized=False):
        '''
        Initialization of variables
        :param Y: numpy array depended variable
        :param X: dictionary of numpy array independed variables
        :param npermutations: int number of permutations
        :param diagonal: boolean, False to delete diagonal from the OLS model
        :return:
        '''
        self.X = X                                  # independent variables: dictionary of numpy.array
        self.Y = Y                                  # dependent variable: dictionary numpy.array
        self.n = Y.values()[0].shape[0]             # number of nodes
        self.npermutations = npermutations          # number of permutations
        self.diagonal = diagonal                    # False then diagonal is removed
        self.directed = directed                    # directed True, undirected False
        self.data = None                            # Pandas DataFrame
        self.model = None                           # OLS Model y ~ x1 + x2 + x3 (original)
        self.v = collections.OrderedDict()          # vectorized matrices, flatten variables with no diagonal
        self.betas = collections.OrderedDict()      # betas distribution
        self.tvalues = collections.OrderedDict()    # t-test values
        self.logfile = logfile                      # logfile path name
        self.standarized = standarized
        self.memory = memory if memory is not None else Profiling()  # to track memory usage

    def init(self):
        '''
        Generating the original OLS model. Y and Xs are flattened.
        Also, the betas and tvalues dictionaries are initialized (key:independent variables, value:[])
        :return:
        '''
        self.v[self.Y.keys()[0]] = self._getFlatten(self.Y.values()[0])
        self._initCoefficients(INTERCEPT)
        for k,x in self.X.items():
            if k == self.Y.keys()[0]:
                utils.printf('ERROR: Idependent variable cannot be named \'[}\''.format(self.Y.keys()[0]), self.logfile)
                sys.exit(0)
            self.v[k] = self._getFlatten(x)
            self._initCoefficients(k)
        self.data = pandas.DataFrame(self.v)
        self.model = self._fit(self.v.keys(), self.data)
        del(self.X)

    def profiling(self, key):
        self.memory.check_memory(key)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def mrqap(self):
        '''
        MultipleRegression Quadratic Assignment Procedure
        :return:
        '''
        directed = 'd' if self.directed else 'i'
        key = self.npermutations if self.memory.perm else self.n
        self.profiling('init-{}-{}'.format(directed, key))
        self.init()
        self.profiling('shuffle-{}-{}'.format(directed, key))
        self._shuffle()
        self.profiling('end-{}-{}'.format(directed, key))

    def _shuffle(self):
        '''
        Shuffling rows and columns npermutations times.
        beta coefficients and tvalues are stored.
        :return:
        '''
        for p in range(self.npermutations):
            self.Ymod = self.Y.values()[0].copy()
            self._rmperm()
            model = self._newfit()
            self._update_betas(model._results.params)
            self._update_tvalues(model.tvalues)
            self.Ymod = None
        gc.collect()


    def _newfit(self):
        '''
        Generates a new OLS fit model
        :return:
        '''
        newv = collections.OrderedDict()
        newv[self.Y.keys()[0]] = self._getFlatten(self.Ymod)
        for k,x in self.v.items():
            if k != self.Y.keys()[0]:
                newv[k] = x
        newdata = pandas.DataFrame(newv)
        newfit = self._fit(newv.keys(), newdata)
        del(newdata)
        del(newv)
        return newfit


    #####################################################################################
    # Handlers
    #####################################################################################

    def _fit(self, keys, data):
        '''
        Fitting OLS model
        v a dictionary with all variables.
        :return:
        '''
        if self.standarized:
            data = data.apply(lambda x: (x - np.mean(x)) / (np.std(x)), axis=0) #axis: 0 to each column, 1 to each row

        formula = '{} ~ {}'.format(self.Y.keys()[0], ' + '.join([k for k in keys if k != self.Y.keys()[0]]))
        return ols(formula, data).fit()

    def _initCoefficients(self, key):
        self.betas[key] = []
        self.tvalues[key] = []

    def _rmperm(self, duplicates=True):
        shuffle = np.random.permutation(self.Ymod.shape[0])
        np.take(self.Ymod,shuffle,axis=0,out=self.Ymod)
        np.take(self.Ymod,shuffle,axis=1,out=self.Ymod)
        del(shuffle)

    def _update_betas(self, betas):
        for idx,k in enumerate(self.betas.keys()):
                self.betas[k].append(round(betas[idx],6))

    def _update_tvalues(self, tvalues):
        for k in self.tvalues.keys():
            self.tvalues[k].append(round(tvalues[k],6))

    def _getFlatten(self, original):
        return self._deleteDiagonalFlatten(original)


    def _deleteDiagonalFlatten(self, original):
        tmp = original.flatten()
        if not self.diagonal:
            tmp = np.delete(tmp, [i*(original.shape[0]+1)for i in range(original.shape[0])])
        return tmp

    def _zeroDiagonalFlatten(self, original):
        tmp = original.copy()
        if not self.diagonal:
            np.fill_diagonal(tmp,0)
        f = tmp.flatten()
        del(tmp)
        return f


    #####################################################################################
    # Prints
    #####################################################################################

    def summary(self):
        '''
        Prints the OLS original summary and beta and tvalue summary.
        :return:
        '''
        self._summary_ols()
        self._summary_betas()
        self._summary_tvalues()
        self._ttest()

    def _summary_ols(self):
        '''
        Print the OLS summary
        :return:
        '''
        utils.printf('', self.logfile)
        utils.printf('=== Summary OLS (original) ===\n{}'.format(self.model.summary()), self.logfile)
        utils.printf('', self.logfile)
        utils.printf('# of Permutations: {}'.format(self.npermutations), self.logfile)

    def _summary_betas(self):
        '''
        Summary of beta coefficients
        :return:
        '''
        utils.printf('', self.logfile)
        utils.printf('=== Summary beta coefficients ===', self.logfile)
        utils.printf('{:20s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.format('INDEPENDENT VAR.','MIN','MEDIAN','MEAN','MAX','STD. DEV.','B.COEFF.','As Large', 'As Small', 'P-VALUE'), self.logfile)
        for k,v in self.betas.items():
            beta = self.model.params[k]
            pstats = self.model.pvalues[k]
            aslarge = sum([1 for c in v if c >= beta]) / float(len(v))
            assmall = sum([1 for c in v if c <= beta]) / float(len(v))
            utils.printf('{:20s}{:10f}{:10f}{:10f}{:10f}{:12f}{:12f}{:12f}{:12f}{:12f}'.format(k,min(v),sorted(v)[len(v)/2],sum(v)/len(v),max(v),round(np.std(v),6),beta,aslarge,assmall,round(float(pstats),2)), self.logfile)

    def _summary_tvalues(self):
        '''
        Summary t-values
        :return:
        '''
        utils.printf('', self.logfile)
        utils.printf('=== Summary T-Values ===', self.logfile)
        utils.printf('{:20s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12s}{:>12s}{:>12s}{:>12s}'.format('INDEPENDENT VAR.','MIN','MEDIAN','MEAN','MAX','STD. DEV.','T-TEST','As Large', 'As Small'), self.logfile)
        for k,v in self.tvalues.items():
            tstats = self.model.tvalues[k]
            aslarge = sum([1 for c in v if c >= tstats]) / float(len(v))
            assmall = sum([1 for c in v if c <= tstats]) / float(len(v))
            utils.printf('{:20s}{:10f}{:10f}{:10f}{:10f}{:12f}{:12f}{:12f}{:12f}'.format(k,min(v),sorted(v)[len(v)/2],sum(v)/len(v),max(v),round(np.std(v),6),round(float(tstats),2),aslarge,assmall), self.logfile)

    def _ttest(self):
        utils.printf('')
        utils.printf('========== T-TEST ==========')
        utils.printf('{:25s} {:25s} {:25s} {:25s}'.format('IND. VAR.','COEF.','T-STAT','P-VALUE'))

        ts = {}
        lines = {}
        for k,vlist in self.betas.items():
            t = stats.ttest_1samp(vlist,self.model.params[k])
            ts[k] = abs(round(float(t[0]),6))
            lines[k] = '{:20s} {:25f} {:25f} {:25f}'.format(k,self.model.params[k],round(float(t[0]),6),round(float(t[1]),6))

        ts = utils.sortDictByValue(ts,True)
        for t in ts:
            utils.printf(lines[t[0]])


    #####################################################################################
    # Plots
    #####################################################################################

    def plot(self,coef='betas',fn=None):
        '''
        Plots frequency of pearson's correlation values
        :param coef: string \in {betas, tvalues}
        :return:
        '''
        ncols = 3
        m = len(self.betas.keys())
        ranges = range(ncols, m, ncols)
        i = np.searchsorted(ranges, m, 'left')
        nrows = len(ranges)

        if i == nrows:
            ranges.append((i+1)*ncols)
            nrows += 1

        fig = plt.figure(figsize=(8,3*i))
        for idx,k in enumerate(self.betas.keys()):
            plt.subplot(nrows,ncols,idx+1)

            if coef == 'betas':
                plt.hist(self.betas[k])
            elif coef == 'tvalues':
                plt.hist(self.tvalues[k])

            plt.xlabel('regression coefficients', fontsize=8)
            plt.ylabel('frequency', fontsize=8)
            plt.title(k)
            plt.grid(True)

        for ax in fig.get_axes():
            ax.tick_params(axis='x', labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        plt.tight_layout()
        plt.savefig(fn)
        plt.close()







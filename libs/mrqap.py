__author__ = 'espin'

#######################################################################
# Dependencies
#######################################################################
import sys
import collections
import numpy as np
import pandas
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from libs import utils


#######################################################################
# MRQAP
#######################################################################
INTERCEPT = 'Intercept'

class MRQAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, X=None, npermutations=-1, diagonal=False):
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
        self.npermutations = npermutations          # number of permutations
        self.diagonal = diagonal                    # False then diagonal is removed
        self.data = None                            # Pandas DataFrame
        self.model = None                           # OLS Model y ~ x1 + x2 + x3 (original)
        self.v = None                               # vectorized matrices, flatten variables with no diagonal
        self.betas = collections.OrderedDict()      # betas distribution
        self.tvalues = collections.OrderedDict()    # t-test values
        self._perms = []                            # list to keep track of already executed permutations

    def init(self):
        '''
        Generating the original OLS model. Y and Xs are flattened.
        Also, the betas and tvalues dictionaries are initialized (key:independent variables, value:[])
        :return:
        '''
        self.v = {self.Y.keys()[0]:self._getFlatten(self.Y.values()[0])}
        self._initCoefficients(INTERCEPT)
        for k,x in self.X.items():
            if k == self.Y.keys()[0]:
                utils.printf('ERROR: Idependent variable cannot be named \'[}\''.format(self.Y.keys()[0]))
                sys.exit(0)
            self.v[k] = self._getFlatten(x)
            self._initCoefficients(k)
        self.data = pandas.DataFrame(self.v)
        self.model = self._fit(self.v, self.data)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def mrqap(self):
        '''
        MultipleRegression Quadratic Assignment Procedure
        :return:
        '''
        self.init()
        self._shuffle()

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

    def _newfit(self):
        '''
        Generates a new OLS fit model
        :return:
        '''
        newv = {self.Y.keys()[0]:self._getFlatten(self.Ymod)}
        for k,x in self.v.items():
            if k != self.Y.keys()[0]:
                newv[k] = x
        newdata = pandas.DataFrame(newv)
        return self._fit(newv, newdata)


    #####################################################################################
    # Handlers
    #####################################################################################

    def _fit(self, v, data):
        '''
        Fitting OLS model
        v a dictionary with all variables.
        :return:
        '''
        return ols('{} ~ {}'.format(self.Y.keys()[0], ' + '.join([k for k in v.keys() if k != self.Y.keys()[0]])), data).fit()

    def _initCoefficients(self, key):
        self.betas[key] = []
        self.tvalues[key] = []

    def _rmperm(self, duplicates=False):
        shuffle = np.random.permutation(self.Ymod.shape[0])
        if not duplicates:
            while list(shuffle) in self._perms:
                shuffle = np.random.permutation(self.Ymod.shape[0])
            self._perms.append(list(shuffle))
        np.take(self.Ymod,shuffle,axis=0,out=self.Ymod)
        np.take(self.Ymod,shuffle,axis=1,out=self.Ymod)

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

    def _summary_ols(self):
        '''
        Print the OLS summary
        :return:
        '''
        utils.printf('')
        utils.printf('=== Summary OLS (original) ===\n{}'.format(self.model.summary()))
        utils.printf('')
        utils.printf('# of Permutations: {}'.format(self.npermutations))

    def _summary_betas(self):
        '''
        Summary of beta coefficients
        :return:
        '''
        utils.printf('')
        utils.printf('=== Summary beta coefficients ===')
        utils.printf('{:20s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.format('INDEPENDENT VAR.','MIN','MEDIAN','MEAN','MAX','STD. DEV.','B.COEFF.','As Large', 'As Small', 'P-VALUE'))
        for k,v in self.betas.items():
            beta = self.model.params[k]
            pstats = self.model.pvalues[k]
            aslarge = sum([1 for c in v if c >= beta]) / float(len(v))
            assmall = sum([1 for c in v if c <= beta]) / float(len(v))
            utils.printf('{:20s}{:10f}{:10f}{:10f}{:10f}{:12f}{:12f}{:12f}{:12f}{:12f}'.format(k,min(v),sorted(v)[len(v)/2],sum(v)/len(v),max(v),round(np.std(v),6),beta,aslarge,assmall,round(float(pstats),2)))

    def _summary_tvalues(self):
        '''
        Summary t-values
        :return:
        '''
        utils.printf('')
        utils.printf('=== Summary T-Values ===')
        utils.printf('{:20s}{:>10s}{:>10s}{:>10s}{:>10s}{:>12s}{:>12s}{:>12s}{:>12s}'.format('INDEPENDENT VAR.','MIN','MEDIAN','MEAN','MAX','STD. DEV.','T-TEST','As Large', 'As Small'))
        for k,v in self.tvalues.items():
            tstats = self.model.tvalues[k]
            aslarge = sum([1 for c in v if c >= tstats]) / float(len(v))
            assmall = sum([1 for c in v if c <= tstats]) / float(len(v))
            utils.printf('{:20s}{:10f}{:10f}{:10f}{:10f}{:12f}{:12f}{:12f}{:12f}'.format(k,min(v),sorted(v)[len(v)/2],sum(v)/len(v),max(v),round(np.std(v),6),round(float(tstats),2),aslarge,assmall))


    #####################################################################################
    # Plots
    #####################################################################################

    def plot(self,coef='betas'):
        '''
        Plots frequency of pearson's correlation values
        :param coef: string \in {betas, tvalues}
        :return:
        '''
        plt.figure(1)
        ncols = round(len(self.betas.keys())/2.0)
        for idx,k in enumerate(self.betas.keys()):
            plt.subplot(2,ncols,idx+1)
            if coef == 'betas':
                plt.hist(self.betas[k])
            elif coef == 'tvalues':
                plt.hist(self.tvalues[k])
            plt.xlabel('regression coefficients')
            plt.ylabel('frequency')
            plt.title(k)
            plt.grid(True)
        plt.suptitle('{} Distribution'.format(coef.upper()))
        plt.show()
        plt.close()
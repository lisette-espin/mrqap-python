__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from libs import utils


#######################################################################
# QAP
#######################################################################
class QAP():

    #####################################################################################
    # Constructor and Init
    #####################################################################################

    def __init__(self, Y=None, X=None, npermutations=-1, diagonal=False):
        '''
        Initialization of variables
        :param Y: numpy array depended variable
        :param X: numpy array independed variable
        :return:
        '''
        self.Y = Y
        self.X = X
        self.npermutations = npermutations
        self.diagonal = diagonal
        self.beta = None
        self.Ymod = None
        self.betas = []

    def init(self):
        '''
        Shows the correlation of the initial/original variables (no shuffeling)
        :return:
        '''
        self.beta = self.correlation(self.X, self.Y)
        self.stats(self.X, self.Y)

    #####################################################################################
    # Core QAP methods
    #####################################################################################

    def qap(self):
        '''
        Quadratic Assignment Procedure
        :param npermutations:
        :return:
        '''
        self.init()
        self._shuffle()

    def _shuffle(self):
        self.Ymod = self.Y.copy()
        for t in range(self.npermutations):
            self._rmperm()
            self._addBeta(self.correlation(self.X, self.Ymod, False))

    def correlation(self, x, y, show=True):
        '''
        Computes Pearson's correlation value of variables x and y.
        Diagonal values are removed.
        :param x: numpy array independent variable
        :param y: numpu array dependent variable
        :param show: if True then shows pearson's correlation and p-value.
        :return:
        '''
        if not self.diagonal:
            xflatten = np.delete(x, [i*(x.shape[0]+1)for i in range(x.shape[0])])
            yflatten = np.delete(y, [i*(y.shape[0]+1)for i in range(y.shape[0])])
            pc = pearsonr(xflatten, yflatten)
        else:
            pc = pearsonr(x.flatten(), y.flatten())
        if show:
            utils.printf('Pearson Correlation: {}'.format(pc[0]))
            utils.printf('p-value: {}'.format(pc[1]))
        return pc

    #####################################################################################
    # Handlers
    #####################################################################################

    def _addBeta(self, p):
        '''
        frequency dictionary of pearson's correlation values
        :param p: person's correlation value
        :return:
        '''
        p = round(p[0],6)
        self.betas.append(p)

    def _rmperm(self):
        shuffle = np.random.permutation(self.Ymod.shape[0])
        np.take(self.Ymod,shuffle,axis=0,out=self.Ymod)
        np.take(self.Ymod,shuffle,axis=1,out=self.Ymod)


    #####################################################################################
    # Plots & Prints
    #####################################################################################

    def summary(self):
        utils.printf('')
        utils.printf('# Permutations: {}'.format(self.npermutations))
        utils.printf('Correlation coefficients: Obs. Value({}), Significance({})'.format(self.beta[0], self.beta[1]))
        utils.printf('')
        utils.printf('- Sum all betas: {}'.format(sum(self.betas)))
        utils.printf('- Min betas: {}'.format(min(self.betas)))
        utils.printf('- Max betas: {}'.format(max(self.betas)))
        utils.printf('- Average betas: {}'.format(np.average(self.betas)))
        utils.printf('- Std. Dev. betas: {}'.format(np.std(self.betas)))
        utils.printf('')
        utils.printf('prop >= {}: {}'.format(self.beta[0], sum([1 for b in self.betas if b >= self.beta[0] ])/float(len(self.betas))))
        utils.printf('prop <= {}: {} (proportion of randomly generated correlations that were as {} as the observed)'.format(self.beta[0], sum([1 for b in self.betas if b <= self.beta[0] ])/float(len(self.betas)), 'large' if self.beta[0] >= 0 else 'small'))
        utils.printf('')

    def plot(self):
        '''
        Plots frequency of pearson's correlation values
        :return:
        '''
        plt.hist(self.betas)
        plt.xlabel('regression coefficients')
        plt.ylabel('frequency')
        plt.title('QAP')
        plt.grid(True)
        plt.show()
        plt.close()

    #####################################################################################
    # Others
    #####################################################################################

    def stats(self, x, y):
        if not self.diagonal:
            xflatten = np.delete(x, [i*(x.shape[0]+1)for i in range(x.shape[0])])
            yflatten = np.delete(y, [i*(y.shape[0]+1)for i in range(y.shape[0])])
            p = np.corrcoef(xflatten,yflatten)
            utils.printf('Pearson\'s correlation:\n{}'.format(p))
            utils.printf('Z-Test:{}'.format(ztest(xflatten, yflatten)))
            utils.printf('T-Test:{}'.format(ttest_ind(xflatten, yflatten)))
        else:
            p = np.corrcoef(x, y)
            utils.printf('Pearson\'s correlation:\n{}'.format(p))
            utils.printf('Z-Test:{}'.format(ztest(x, y)))
            utils.printf('T-Test:{}'.format(ttest_ind(x, y)))

    def ols(self, x, y):
        xflatten = np.delete(x, [i*(x.shape[0]+1)for i in range(x.shape[0])])
        yflatten = np.delete(y, [i*(y.shape[0]+1)for i in range(y.shape[0])])
        xflatten = sm.add_constant(xflatten)
        model = sm.OLS(yflatten,xflatten)
        results = model.fit()
        print(results.summary())

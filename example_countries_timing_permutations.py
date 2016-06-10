__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
from libs.mrqap import MRQAP
import time
from libs import utils
from libs.profiling import Profiling
import sys


#######################################################################
# Functions
#######################################################################
NCOUNTRIES = 249
def getMatrix(path, directed=False, log1p=False):
    matrix = np.zeros(shape=(NCOUNTRIES,NCOUNTRIES))
    with open(path, 'rb') as f:
        for line in f:
            data = line.split(' ')
            c1 = int(data[0])-1
            c2 = int(data[1])-1
            v = np.log1p(float(data[2])) if log1p else float(data[2])
            matrix[c1][c2] = v # real data from file
            if not directed:
                matrix[c2][c1] = v # symmetry
    print '{} loaded as a matrix!'.format(path)
    return matrix

#######################################################################
# Main
######################################################################
@profile
def main(directed):

    logfile = 'results-permutations/timigs-{}.txt'.format('directed' if directed else 'undirected')
    memory = Profiling('Permutations {}'.format('directed' if directed else 'undirected'), 'results-permutations/python-profiling-nperm-nodes{}-{}.png'.format(NCOUNTRIES,'directed' if directed else 'undirected'), True)
    memory.check_memory('init-{}'.format('d' if directed else 'i'))

    #######################################################################
    # Data Matrices
    #######################################################################
    X1 = getMatrix('data-permutations/country_trade_index.txt',directed,True)
    memory.check_memory('X1-{}'.format('d' if directed else 'i'))
    X2 = getMatrix('data-permutations/country_distance_index.txt',directed,True)
    memory.check_memory('X2-{}'.format('d' if directed else 'i'))
    X3 = getMatrix('data-permutations/country_colonial_index.txt',directed)
    Y  = getMatrix('data-permutations/country_lang_index.txt',directed)
    memory.check_memory('Y-{}'.format('d' if directed else 'i'))
    X = {'TRADE':X1, 'DISTANCE':X2, 'COLONIAL':X3}
    Y = {'LANG':Y}
    np.random.seed(1)

    #######################################################################
    # QAP
    #######################################################################
    perms = np.logspace(1,7,num=7-1, endpoint=False)
    for nperm in perms:
        start_time = time.time()
        mrqap = MRQAP(Y=Y, X=X, npermutations=int(nperm), diagonal=False, directed=directed, logfile=logfile, memory=memory)
        mrqap.mrqap()

        utils.printf("--- {}, nperm {}: {} seconds ---".format('directed' if directed else 'undirected', nperm, time.time() - start_time), logfile)
        mrqap.summary()

        fn = 'results-permutations/python-nperm{}-{}-<coef>.png'.format(nperm,'directed' if directed else 'undirected')
        mrqap.plot('betas', fn.replace('<coef>','betas'))
        mrqap.plot('tvalues', fn.replace('<coef>','tvalues'))

        utils.printf('******************************************************************************\n\n', logfile)
        del(mrqap)
    return


if __name__ == '__main__':
    directed = sys.argv[1] == '1'
    main(directed)
    sys.exit(0)

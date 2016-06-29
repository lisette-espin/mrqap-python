__author__ = 'lisette.espin'

#######################################################################
# Dependencies
#######################################################################
import numpy as np
from libs.mrqap import MRQAP
import time
import networkx as nx
from libs import utils
from libs.profiling import Profiling
import threading
import time
import sys
from libs.mtTkinter import *
import os

#######################################################################
# Constants
#######################################################################
NPERMUTATIONS = 1000
EDGEPROB = 0.1
SEED = 1
np.random.seed(SEED)

#######################################################################
# Global
#######################################################################


#######################################################################
# Functions
#######################################################################
def generateGraph(nnodes, edgeprob, directed, pathtosave):
    if os.path.exists(pathtosave):
        matrix = np.loadtxt(pathtosave)
    else:
        shape = (nnodes,nnodes)
        G = nx.fast_gnp_random_graph(n=nnodes, p=edgeprob, directed=directed)
        matrix = nx.adjacency_matrix(G)

        if pathtosave is not None:
            np.savetxt(pathtosave, matrix.toarray(), fmt='%d',)

        print nx.info(G)
        matrix = matrix.toarray()

    return matrix

#######################################################################
# Main
#######################################################################
@profile
def main(directed):

    logfile = 'results-synthetic-ernos-renyi/timigs-{}.txt'.format('directed' if directed else 'undirected')
    memory = Profiling('Nodes {}'.format('directed' if directed else 'undirected'), 'results-synthetic-ernos-renyi/python-profiling-netsize-edgeprob{}-nperm{}-{}.png'.format(EDGEPROB, NPERMUTATIONS,'directed' if directed else 'undirected'), False)
    memory.check_memory('init-{}'.format('d' if directed else 'i'))

    #######################################################################
    # Data Matrices
    #######################################################################
    #nnodes = np.logspace(1,7,num=7-1, endpoint=False)
    nnodes = np.logspace(1,5,num=5-1, endpoint=False)
    for n in nnodes:
        n = int(n)
        fn = 'data-synthetic-ernos-renyi/nodes{}_edgeprob{}_<var>.dat'.format(n,EDGEPROB)
        memory.check_memory('nodes-{}'.format(n))
        X1 = generateGraph(n,EDGEPROB,directed, fn.replace('<var>','X1'))
        memory.check_memory('X1-{}'.format(n))
        X2 = generateGraph(n,EDGEPROB,directed, fn.replace('<var>','X2'))
        memory.check_memory('X2-{}'.format(n))
        X3 = generateGraph(n,EDGEPROB,directed, fn.replace('<var>','X3'))
        memory.check_memory('X3-{}'.format(n))
        Y  = generateGraph(n,EDGEPROB,directed, fn.replace('<var>','Y'))
        memory.check_memory('Y-{}'.format(n))
        X = {'X1':X1, 'X2':X2, 'X3':X3}
        Y = {'Y':Y}

        #######################################################################
        # QAP
        #######################################################################
        start_time = time.time()
        mrqap = MRQAP(Y=Y, X=X, npermutations=int(NPERMUTATIONS), diagonal=False, directed=directed, logfile=logfile, memory=memory)
        mrqap.mrqap()

        utils.printf("\n--- {}, nodes {}: {} seconds ---".format('directed' if directed else 'undirected', n, time.time() - start_time), logfile)
        mrqap.summary()

        fn = 'results-synthetic-ernos-renyi/python-nodes{}-edgeprob{}-nperm{}-{}-<coef>.png'.format(n, EDGEPROB, NPERMUTATIONS,'directed' if directed else 'undirected')
        mrqap.plot('betas',fn.replace('<coef>','betas'))
        mrqap.plot('tvalues',fn.replace('<coef>','tvalues'))

        utils.printf('******************************************************************************\n\n', logfile)
        del(mrqap)
    return


if __name__ == '__main__':
    directed = sys.argv[1] == '1'
    main(directed)
    sys.exit(0)

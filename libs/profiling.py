__author__ = 'espin'

#######################################################################################
### Dependences
### Reference:
### http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/
#######################################################################################
import resource
import psutil
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from libs import utils
import copy
from threading import Thread
import time
from collections import OrderedDict

#######################################################################################
# FUNCTIONS
#######################################################################################

class Profiling():

    def __init__(self, title=None, fn=None, perm=False):
        self.mem_resource = OrderedDict()
        self.mem_psutil = OrderedDict()
        self.cpu_usage = OrderedDict()
        self.virtual_memory_usage = OrderedDict()
        self.swap_memory_usage = OrderedDict()
        self.title = title
        self.fn = fn
        self.perm = perm

    def check_memory(self, key):
        self.memory_usage_resource(key)
        self.memory_usage_psutil(key)
        self.memory_cpu_usage(key)
        self.plot()

    def kill_if_necessary(self, vm, sm):
        if vm.percent >= 85. or sm.percent >= 85.:
            print('FULL MEMORY: \n- Virtual Memory: {}\n- Swap Memory: {}'.format(vm, sm))

    def memory_cpu_usage(self, key):
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        self.cpu_usage[key] = int(255 * cpu / 100)
        self.virtual_memory_usage[key] = vm.percent
        self.swap_memory_usage[key] = sm.percent
        self.kill_if_necessary(vm, sm)


    def memory_usage_psutil(self, key):
        # return the memory usage in MB
        process = psutil.Process(os.getpid())
        self.mem_psutil[key] = process.memory_info()[0] / float(2 ** 20)

    def memory_usage_resource(self, key):
        rusage_denom = 1024.
        if sys.platform == 'darwin':
            # ... it seems that in OSX the output is different units ...
            rusage_denom = rusage_denom * rusage_denom
        self.mem_resource[key] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom

    def copy(self, mem):
        self.mem_resource = mem.mem_resource.deepcopy()
        self.mem_psutil = mem.mem_psutil.deepcopy()
        self.cpu_usage = mem.cpu_usage.deepcopy()
        self.virtual_memory_usage = mem.virtual_memory_usage.deepcopy()
        self.swap_memory_usage = mem.swap_memory_usage.deepcopy()

    def plot(self):
        '''
        Plots the Memory usage in MB
        :return:
        '''
        if self.fn is not None and self.title is not None:
            labels = self.mem_resource.keys()
            x = range(len(labels))

            plt.figure(1)
            ax1 = plt.subplot(211)
            ax1.plot(x, self.mem_resource.values(), color='red', marker='o', label='mem_resource')
            ax1.plot(x, self.mem_psutil.values(), color='blue', marker='o', label='mem_psutil')
            ax1.set_ylabel('Memory usage in MB')
            ax1.grid(True)
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=20, fontsize=7)
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':10})

            ax2 = plt.subplot(212)
            ax2.plot(x, self.cpu_usage.values(), color='black', marker='o', label='cpu_usage')
            ax2.plot(x, self.virtual_memory_usage.values(), color='orange', marker='o', label='virtual_memory')
            ax2.plot(x, self.swap_memory_usage.values(), color='green', marker='o', label='swap_memory')
            #ax2.set_xlabel(self.xlabel)
            ax2.set_ylabel('Percentage (usage)')
            ax2.grid(True)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=20, fontsize=7)
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax2.legend(loc='center left', prop={'size':10}) # bbox_to_anchor=(1, 0.5),

            plt.suptitle('Profiling - {}'.format(self.title))
            plt.tight_layout()
            plt.savefig(self.fn)
            plt.close()

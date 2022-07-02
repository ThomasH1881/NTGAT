import numpy as np
import dgl
import torch
import argparse
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import PubmedGraphDataset
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
from modules import *
import sys

parser = argparse.ArgumentParser()
register_data_args(parser)
args = parser.parse_args()
print(args.dataset)

# load dataset and set parameters
if args.dataset == 'ogbn-arxiv':
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='/home/nfs_data/datasets/')
    g, labels = dataset[0]
    features = g.ndata['feat']
    g = dgl.to_bidirected(g)
    g = g.remove_self_loop().add_self_loop()
else:
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset()
    else:
        print("dataset is not defined")
        sys.exit(0)
    g = data[0]
    features = g.ndata['feat']

num_feats = features.shape[1]
num_nodes = features.shape[0]
n_edges = g.number_of_edges()
len_features = 64

# parameters of hardwares
num_PEs = 64
num_chunks = 256
num_cache = 256
fifo_size = 5
num_heads = 8
DSP_Dense = 4096
DSP_VP = 32
# delay time and MAC time
DU_delay = int(np.log2(num_chunks)) - 2
VP_delay = int(np.log2(num_PEs))-2
VP_time = int(len_features/DSP_VP)
CB_delay = int(np.log2(num_PEs))-2
CB_time = int(len_features/DSP_VP)
#cache_size = 20 * 1024 * 1024 / 4 / num_cache / len_features

'''
def resources():
    num_DSP = 6840
    num_SRAM = 35 * 1024 * 1024
    used_SRAM = num_nodes * 2 * num_heads * 4
    used_DSP = num_PEs  # exponent
    used_DSP += num_PEs * DSP_VP

    print(len_features)
    return
'''

# define all modules
DU = [Decoder_Unit(i, DU_delay) for i in range(num_PEs)]
SSR = [Swap_Shift_Register(i) for i in range(num_PEs)]
VP = [Vector_Processor(i, num_cache, VP_delay, VP_time) for i in range(num_PEs)]
fifo = [e_FIFO(i, fifo_size) for i in range(num_chunks)]
CB = [CacheBlock(i, 320, 5, CB_delay, CB_time) for i in range(num_cache)]
DDR = [DDRModel(i, 5, int(len_features/DSP_VP), 72) for i in range(4)]

# global clock
time = 0   #全局time
time_next = 10000
node_i = 0

print(num_nodes)

# judging if all the nodes have been finished. node_i should be num_nodes, and all modules are idle with no workload under processing
def top():
    if (node_i < num_nodes):
        return True
    else:
        for i in range(num_PEs):
            if (DU[i].idle == False): return True
        for i in range(num_chunks):
            if (not fifo[i].empty()): return True
        for i in range(num_PEs):
            if (SSR[i].idle == False): return True
        for i in range(num_PEs):
            if (VP[i].idle == False): return True
        for i in range(num_cache):
            if (not CB[i].fifo.empty()): return True
        for i in range(4):
            if (not DDR[i].waitlist.empty()): return True
            if (not DDR[i].timelist == []): return True

        return False

# cache statistics.
cache_block_size = 4
cache_size = 256
hmf = [0, 0, 0]
total = [0]

# main iteration

# Time protocol

# 0: if XXX.time_stamp > time, event hasn't finished, do nothing    # future time protocol
# 1: if (XXX.time_stamp < time): XXX.time_stamp = time              # current time protocol
# 2: XXX.step(fifo)
# 3: if (XXX.time_stamp < time_next): time_next = XXX.time_stamp    # next time protocol

# Most modules step in this way. First, its time_stamp should be at least now when stepping.
# after step, its time_stamp must increase. (guaranteed by each step())
# Finally, time_next should be the earliest time_stamp of all modules

while (top()):   # judge if simulation finishes
    for i in range(num_PEs):

        if (DU[i].time_stamp > time):                                          # future time protocol
            if (DU[i].time_stamp < time_next): time_next = DU[i].time_stamp

        elif (DU[i].idle and SSR[i].idle and node_i < num_nodes):           # register DU with a new node
            print("register DU", i, "for node", node_i, "at time:", time)
            nlist=[node_i] + g.find_edges(g.in_edges(node_i, form='eid'))[0].tolist()   # get all neighbours of the node
            DU[i].register(node_i, nlist, time)           # use node index, neighbour list and current time to register
            # the sorting processor was originally called SSR, this simulator continues to use this name.
            SSR[i].register(node_i, (g.in_degrees(node_i)+1), (g.in_degrees(node_i)+1))  # register SSR together
            node_i = node_i + 1
            if (DU[i].time_stamp < time_next): time_next = DU[i].time_stamp   # future time protocol

        elif (DU[i].idle == False):
            if (DU[i].time_stamp < time): DU[i].time_stamp = time              # current time protocol
            DU[i].step(fifo)
            if (DU[i].time_stamp < time_next): time_next = DU[i].time_stamp    # next time protocol
    #print(time,time_next)

    for i in range(num_chunks):
        if (fifo[i].time_stamp > time):
            if (fifo[i].time_stamp < time_next): time_next = fifo[i].time_stamp   # future time protocol
        elif (not fifo[i].empty()):
            if (fifo[i].time_stamp < time): fifo[i].time_stamp = time             # current time protocol
            fifo[i].step(SSR)
            #print("fifo", i ,"step")
            if (fifo[i].time_stamp < time_next): time_next = fifo[i].time_stamp   # next time protocol
    #print(time,time_next)

    for i in range(num_PEs):
        if (SSR[i].time_stamp > time):                                            # future time protocol
            if (SSR[i].time_stamp < time_next): time_next = SSR[i].time_stamp
        elif (SSR[i].idle == False and SSR[i].count == SSR[i].degree and VP[i].idle): #SSR not full or VP is working, do nothing
            if (SSR[i].time_stamp < time): SSR[i].time_stamp = time               # current time protocol
            SSR[i].step(VP, total)
            #print("SSR", i ,"step")
            if (SSR[i].time_stamp < time_next): time_next = SSR[i].time_stamp     # next time protocol
    #print(time, time_next)

    for i in range(num_PEs):
        if (VP[i].time_stamp > time):
            if (VP[i].time_stamp < time_next): time_next = VP[i].time_stamp       # future time protocol
        elif (VP[i].idle == False):
            if (VP[i].time_stamp < time): VP[i].time_stamp = time                 # current time protocol
            VP[i].step(CB, hmf)
            if (VP[i].time_stamp < time_next): time_next = VP[i].time_stamp       # next time protocol
    #print(time, time_next)

    for i in range(num_cache):
        if (CB[i].time_stamp > time):
            if (CB[i].time_stamp < time_next): time_next = CB[i].time_stamp       # future time protocol
        elif (not CB[i].fifo.empty() or CB[i].fetch_start or CB[i].AXI_return):   # judge if CB should step
            if (CB[i].time_stamp < time): CB[i].time_stamp = time                 # current time protocol
            CB[i].step(VP, DDR, hmf)
            #print("CB", i ,"step")
            if (CB[i].time_stamp < time_next): time_next = CB[i].time_stamp       # next time protocol
    #print(time, time_next)

    for i in range(4):
        if (DDR[i].time_stamp > time):
            if (DDR[i].time_stamp < time_next): time_next = DDR[i].time_stamp     # future time protocol
        elif (not DDR[i].waitlist.empty() or DDR[i].timelist != []):              # judge if DDR should step
            if (DDR[i].time_stamp < time): DDR[i].time_stamp = time               # current time protocol
            DDR[i].step(CB)
            #print("DDR", i ,"step")
            if (DDR[i].time_stamp < time_next): time_next = DDR[i].time_stamp     # next time protocol
    #print(time, time_next)

    # errors:
    if (time_next == time):       # a module's time_stamp doesn't change after step
        print("Error at time:", time, "time_next equals time!")
        break
    if (time_next == time + 10000):   # no time_stamp in the future is observed, time_next is undetermined
        for i in range(num_PEs):
            print(SSR[i].idle, SSR[i].count, SSR[i].degree)
        for i in range(num_PEs):
            print(VP[i].idle, VP[i].finished, VP[i].depth)
        print("Error at time:", time, "no step happened")
        break

    # iterate by time_next
    time = time_next
    time_next = time + 10000   # simply initialize time_next with a large time, if will be updated by module's time_stamp
print(time)       # final results of cycles
print(total[0])   # total nodes processed
print(hmf[0],hmf[1],hmf[2])  # final miss/hit/fetch numbers

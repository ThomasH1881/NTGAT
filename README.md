# NTGAT

This repository is the project of Node Tailoring GAT accelerator.


# Dgl Modification Codes:

add my_GATConv function in dgl/nn/pytorch/conv/gatconv.py, see modifications from line 635 to line 683.

add my_edge_softmax function in dgl/ops/edge_softmax.py, see modifications from line 154 to line 246.

These modifications are used for accuracy tests of NTGAT.


# GPU experiments:

run gat/train.py for experiments on Cora/Citeseer/Pubmed/Reddit datasets.

run ogb/ogbn-arxiv/gat.py for experiment on ogbn-arxiv dataset.

  

After GPU running, my_edge_softmax will generate traces. (set saving directory at dgl/ops/edge_softmax.py line 186,187)

Move traces to gat/'simulation with traces', and run gat/'simulation with traces'/main.py for simulation with node tailoring.

  

gat/'simulation with traces'/main.py is the same with Simulator/main.py except for loading different data, annotation can be find in Simulator/main.py.


# Simulator:

Simulator/modules.py defines hardware components of NTGAT.

Simulator/main.py is the primary simulator program.

This simulator runs origin datasets without tailoring.

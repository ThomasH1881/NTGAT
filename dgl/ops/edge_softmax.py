"""dgl edge_softmax operator module."""
from ..backend import edge_softmax as edge_softmax_internal
from ..backend import edge_softmax_hetero as edge_softmax_hetero_internal
from ..backend import astype
from ..base import ALL, is_all

__all__ = ['edge_softmax']


def edge_softmax(graph, logits, eids=ALL, norm_by='dst'):
    r"""Compute softmax over weights of incoming edges for every node.

    For a node :math:`i`, edge softmax is an operation that computes

    .. math::
      a_{ij} = \frac{\exp(z_{ij})}{\sum_{j\in\mathcal{N}(i)}\exp(z_{ij})}

    where :math:`z_{ij}` is a signal of edge :math:`j\rightarrow i`, also
    called logits in the context of softmax. :math:`\mathcal{N}(i)` is
    the set of nodes that have an edge to :math:`i`.

    By default edge softmax is normalized by destination nodes(i.e. :math:`ij`
    are incoming edges of `i` in the formula above). We also support edge
    softmax normalized by source nodes(i.e. :math:`ij` are outgoing edges of
    `i` in the formula). The former case corresponds to softmax in GAT and
    Transformer, and the latter case corresponds to softmax in Capsule network.
    An example of using edge softmax is in
    `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ where
    the attention weights are computed with this operation.
    Other non-GNN examples using this are
    `Transformer <https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`__,
    `Capsule <https://arxiv.org/pdf/1710.09829.pdf>`__, etc.

    Parameters
    ----------
    graph : DGLGraph
        The graph over which edge softmax will be performed.
    logits : torch.Tensor or dict of torch.Tensor
        The input edge feature. Heterogeneous graphs can have dict of tensors where
        each tensor stores the edge features of the corresponding relation type.
    eids : torch.Tensor or ALL, optional
        The IDs of the edges to apply edge softmax. If ALL, it will apply edge
        softmax to all edges in the graph. Default: ALL.
    norm_by : str, could be `src` or `dst`
        Normalized by source nodes or destination nodes. Default: `dst`.

    Returns
    -------
    Tensor or tuple of tensors
        Softmax value.

    Notes
    -----
        * Input shape: :math:`(E, *, 1)` where * means any number of
          additional dimensions, :math:`E` equals the length of eids.
          If the `eids` is ALL, :math:`E` equals the number of edges in
          the graph.
        * Return shape: :math:`(E, *, 1)`

    Examples on a homogeneous graph
    -------------------------------
    The following example uses PyTorch backend.

    >>> from dgl.nn.functional import edge_softmax
    >>> import dgl
    >>> import torch as th

    Create a :code:`DGLGraph` object and initialize its edge features.

    >>> g = dgl.graph((th.tensor([0, 0, 0, 1, 1, 2]), th.tensor([0, 1, 2, 1, 2, 2])))
    >>> edata = th.ones(6, 1).float()
    >>> edata
        tensor([[1.],
                [1.],
                [1.],
                [1.],
                [1.],
                [1.]])

    Apply edge softmax over g:

    >>> edge_softmax(g, edata)
        tensor([[1.0000],
                [0.5000],
                [0.3333],
                [0.5000],
                [0.3333],
                [0.3333]])

    Apply edge softmax over g normalized by source nodes:

    >>> edge_softmax(g, edata, norm_by='src')
        tensor([[0.3333],
                [0.3333],
                [0.3333],
                [0.5000],
                [0.5000],
                [1.0000]])

    Apply edge softmax to first 4 edges of g:

    >>> edge_softmax(g, edata[:4], th.Tensor([0,1,2,3]))
        tensor([[1.0000],
                [0.5000],
                [1.0000],
                [0.5000]])


    Examples on a heterogeneous graph
    ---------------------------------

    Create a heterogeneous graph and initialize its edge features.

    >>> hg = dgl.heterograph({
    ...     ('user', 'follows', 'user'): ([0, 0, 1], [0, 1, 2]),
    ...     ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    ...     })
    >>> edata_follows = th.ones(3, 1).float()
    >>> edata_develops = th.ones(2, 1).float()
    >>> edata_dict = {('user', 'follows', 'user'): edata_follows,
    ... ('developer','develops', 'game'): edata_develops}

    Apply edge softmax over hg normalized by source nodes:

    >>> edge_softmax(hg, edata_dict, norm_by='src')
        {('developer', 'develops', 'game'): tensor([[1.],
        [1.]]), ('user', 'follows', 'user'): tensor([[0.5000],
        [0.5000],
        [1.0000]])}
    """
    if not is_all(eids):
        eids = astype(eids, graph.idtype)
    if graph._graph.number_of_etypes() == 1:
        return edge_softmax_internal(graph._graph, logits,
                                     eids=eids, norm_by=norm_by)
    else:
        logits_list = [None] * graph._graph.number_of_etypes()
        for rel in graph.canonical_etypes:
            etid = graph.get_etype_id(rel)
            logits_list[etid] = logits[rel]
        logits_tuple = tuple(logits_list)
        score_tuple = edge_softmax_hetero_internal(graph._graph,
                                                   eids, norm_by, *logits_tuple)
        score = {}
        for rel in graph.canonical_etypes:
            etid = graph.get_etype_id(rel)
            score[rel] = score_tuple[etid]
        return score


import numpy
import torch
import random
def my_edge_softmax(graph, logits, lowerbound, trunc_k = 0.9, eids=ALL, norm_by='dst'):
    tmp = 0
    #print(logits.size())           # logits tensor stores the attention coefficients.

    print("my_edge_softmax running")
    print("lowerbound:", lowerbound)
    print("trunc_k:", trunc_k)
    
    # fullgraph and truncgraph store the trace of fetching each node, they are saved for simulator.
    #fullgraph = []
    #truncgraph = []
    
    for i in range(graph.num_nodes()):
        d = len(graph.in_edges(i, form='eid'))
        n = trunc(d, lowerbound, trunc_k)
        index = graph.in_edges(i, form='eid').type(torch.int64)      # index of edges in logits matrix
        
        
        #tensor = logits[index].sum(dim=1)
        tensor = logits[index].sum(dim=1).squeeze()             # depending on the dimension, logits may need squeeze()
        argsort = tensor.argsort()                              # sort the tensor
        mask = argsort[0:n]                                     # tailor n attention coefficients

        #mask = random.sample(range(0,d),n)                     # random sampling
        
        logits[index[mask]] = float('-inf')                     # here we set the tailored attention coefficients to -inf, and they will become zero after softmax.
                
        #truncgraph.append(graph.find_edges(graph.in_edges(i, form='eid')[argsort[n:]])[0].cpu().tolist())     # add the trace to truncgraph and fullgraph
        #fullgraph.append(graph.find_edges(graph.in_edges(i, form='eid'))[0].cpu().tolist())
        tmp += n                                                # just count the sum of tailored degrees
    
    #if trunc_k == 0.675:                                       # save the traces of a layer
        #numpy.save('/home/nfs_data/hwt/gat/saved/arxiv2_full',fullgraph)
        #numpy.save('/home/nfs_data/hwt/gat/saved/arxiv2_trunc',truncgraph)
    
    
    # some test codes for dynamic tailoring, it doesn't work very well in test.
    '''
    for i in range(graph.num_nodes()):
        index = graph.in_edges(i, form='eid').type(torch.int64)
        tensor = logits[index].squeeze()
        thr = 0.1 * tensor[tensor.argmax()]
        tensor = tensor - thr
        mask = (tensor < 0).nonzero()
        tensor = tensor + thr
        tmp += len(mask)
        logits[index[mask]] = float('-inf')
    '''
    print(tmp)
    
    # below is the same as original edgesoftmax
    
    if not is_all(eids):
        eids = astype(eids, graph.idtype)
    if graph._graph.number_of_etypes() == 1:
        return edge_softmax_internal(graph._graph, logits,
                                     eids=eids, norm_by=norm_by)
    else:
        logits_list = [None] * graph._graph.number_of_etypes()
        for rel in graph.canonical_etypes:
            etid = graph.get_etype_id(rel)
            logits_list[etid] = logits[rel]
        logits_tuple = tuple(logits_list)
        score_tuple = edge_softmax_hetero_internal(graph._graph,
                                                   eids, norm_by, *logits_tuple)
        score = {}
        for rel in graph.canonical_etypes:
            etid = graph.get_etype_id(rel)
            score[rel] = score_tuple[etid]
        return score
        
'''
# 0.1x + 0.9n = 200     x = 2000 - 9n                
def trunc(n):
    if n < 400:
        return 0
    #elif n < 110:
        #return int((n - 10) * 0.9)
    else:
        return int((n - 400) * 0.8)
'''

# Tailoring function

# 0.1x + 0.9n = 200     x = 2000 - 9n                
def trunc(n, lowerbound = 5, trunc_k = 0.9):
    upbound = 1000
    if n < lowerbound:
        return 0
    elif n < (upbound - trunc_k * lowerbound) / (1-trunc_k):
        return int((n - lowerbound) * trunc_k)
    else:
        return int(n-upbound)
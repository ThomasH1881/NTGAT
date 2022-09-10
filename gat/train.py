"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, AmazonCoBuyComputerDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from ogb.graphproppred import DglGraphPropPredDataset
from dgl.dataloading import GraphDataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from gat import GAT, my_GAT
from utils import EarlyStopping


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        '''
        with profile(
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], with_stack=True, record_shapes=True, use_cuda=True
        )as p:
        
            for iter in range(20):
                logits = model(features)
                p.step()
        p.export_chrome_trace('reddit.json')
        print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1))
        '''
        
        
        #tmp = torch.zeros(features.shape).cuda()
        #for i in range(50):
            #logits = model(tmp)
        #torch.cuda.synchronize()
        time_start_testing = time.time()
        #for i in range(50):
        logits = model(features)
        #torch.cuda.synchronize()
        time_end_testing = time.time()
        print("Inference time:", time_end_testing-time_start_testing)
        
        for i in range(50):
            logits = model(features)
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

def _collate_fn(batch):
    # batch is a list of tuple (graph, label)
    graphs = [e[0] for e in batch]
    g = dgl.batch(graphs)
    labels = [e[1] for e in batch]
    labels = torch.stack(labels, 0)
    return g, labels

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset(raw_dir='/home/nfs_data/datasets/')
    elif args.dataset == 'ACBC':
        data = AmazonCoBuyComputerDataset(raw_dir='/home/nfs_data/datasets/')
    elif args.dataset == 'CCS':
        data = CoauthorCSDataset(raw_dir='/home/nfs_data/datasets/')
    elif args.dataset == 'CPh':
        data = CoauthorPhysicsDataset(raw_dir='/home/nfs_data/datasets/')
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
  
    
    g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_labels
    #n_edges = data.graph.number_of_edges()
    n_edges = g.num_edges()
    n_nodes = g.num_nodes()
    

    '''
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))
    '''
        
    
    
    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    
    '''
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))
    
    #if args.early_stop:
        #model.load_state_dict(torch.load('es_checkpoint.pt'))
    
    torch.save(model.state_dict(),'checkpoint.pt')
    '''
    
    model.load_state_dict(torch.load('checkpoint.pt'))
    acc = evaluate(model, features, labels, test_mask)

    print("Test Accuracy {:.4f}".format(acc))
    model = my_GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual,
                args.lb,
                args.k)
    model.load_state_dict(torch.load('checkpoint.pt'))
    if cuda:
        model.cuda()
    #acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,          # 2 for Reddit
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,         # 64 for Reddit
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,        # 0 for Reddit
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.6,      # 0 for Reddit
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument("--lb", type=int, default=100,
                        help="lowerbound threshold")
    parser.add_argument("--k", type=float, default=0.9,
                        help="trunck rate k")
    args = parser.parse_args()
    print(args)

    main(args)

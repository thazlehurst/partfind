""" PartGNN """

import torch
import dgl
import os
from ABCSiameseDataset import ABCSiameseDataset
from utils import print_graph, compare_graphlets
from step2image_pyocc import render_step

class PartGNN(torch.nn.Module):

    def __init__(self, args):
        """
        :param args: Arguments object
        """
        super(PartGNN, self).__init__()
        self.args = args
        self.verbose = self.args.verbose
        if self.verbose:
            print("Verbose enabled")
        
        
        self.load_dataset()
        #self.setup_layers()
    
    def load_dataset(self):
        # load dataset
        print("Initalising dataset...")
        raw_dir = self.args.dataset
        
        # to impliment load dataset from abc folder
        single_dataset = []
        
        self.dataset = ABCSiameseDataset(
        single_dataset,
        raw_dir=raw_dir,
        verbose=self.verbose)
        ## test
        print("test point")
        for i in range(0,1):
            print("i:",i)
            graphs, targets, filename = self.dataset[i]
            #print("graphs", graphs)
            print("filename:",filename)
            model_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data"
            model_file = os.path.join(model_folder,filename)
            image1 = render_step(model_file)
            print("targets",targets)
            g1,g2,g3 = dgl.unbatch(graphs)
            print("g1",g1)
            print("g2",g2)
            print("g3",g3)
            print_graph(g1,str(i)+"g1.png")
            print_graph(g2,str(i)+"g2.png")
            print_graph(g3,str(i)+"g3.png")
            values2 = compare_graphlets([g1,g2,g3])
            print("values2",values2)

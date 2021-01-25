""" PartGNN """

import torch
import dgl
import os
from Dataset.ABCSiameseDataset import ABCSiameseDataset
from Dataset.ABCDataset import ABCDataset
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
        
        self.model_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data"
        self.step_dataset = []
        
        
        self.create_dataset()
        self.load_dataset()
        #self.setup_layers()
    
    def create_dataset(self):
        print("Loading dataset from step files...")
        model_folder = self.model_folder
        self.step_dataset = ABCDataset(raw_dir='Dataset/')
        
    
    def load_dataset(self):
        # load dataset
        print("Initalising dataset...")
        raw_dir = self.args.dataset
        
        # to impliment load dataset from abc folder
        single_dataset = []
        model_folder = self.model_folder
        self.dataset = ABCSiameseDataset(
        self.step_dataset,
        raw_dir=raw_dir,
        verbose=self.verbose,
        dataset_range=[0,50],force_reload=False)
        ## test
        print("test point")
        for i in range(6,7):
            print("i:",i)
            graphs, targets, filenames = self.dataset[i]
            #print("graphs", graphs)
            print("filename:",filenames)
            #model_folder = self.model_folder
            for filename in filenames:
                model_file = self.gz2step(os.path.join(model_folder,filename))
                image1 = render_step(model_file,remove_tmp=False)
            print("targets",targets)
            g1,g2,g3 = dgl.unbatch(graphs)
            #print("g1",g1)
            #print("g2",g2)
            #print("g3",g3)
            print_graph(g1,str(i)+"g1.png")
            print_graph(g2,str(i)+"g2.png")
            print_graph(g3,str(i)+"g3.png")
            values2, _ = compare_graphlets([g1,g2,g3])
            print("values2",values2)

    def gz2step(self,filename):
        return os.path.splitext(filename)[0]+".step"
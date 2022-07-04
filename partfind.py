# python class for partfind, load this then all the functionality should be in here.
# TH 03/2022


# partgnn isn't in this version, updating from deep cad graph
from CADDataset import CADDataset
from args import parameter_parser
from partgnn import PartGNN
from CreateDataset import create_dataset
import numpy as np
import os

class PartFind():
    def __init__(self):
        
        self.dataset_file = None
        self.dataset = None
        self.args = parameter_parser()
        super(PartFind,self).__init__()
        
        self.model_loaded = False
        self.PGNN = []
        
        
    '''
    Dataset management
    '''
        
        
    def load_dataset(self):
        self.dataset = CADDataset(".\\Dataset\\Dataset-Cakebox", "cakebox_nx.pickle", force_reprocess=False)
        
    def create_dataset_pkl(self,root='.\\Dataset\\Dataset-Mturk',file_name= "abc_abc_dataset.pkl",triple_file=None,add_cats=False):
        '''
        This generates a dataset from .pickled set of nx graphs
        root: The folder your pickle is in and where you want your dataset stored.
        file_name: name of pickle
        force_reprocess: Re processed dataset
        triple_file: is a cvs file containing predetermined triples of file names, a base model, a similar model and a dissimilar model
        add_cats: if true, then category data is used to find similar or dissimilar models
        '''
        print("triple_file",triple_file)
        print("add_cats",add_cats)
        
        if (triple_file == None) and (add_cats == False):
            print("At least one of 'triple_file' or 'add_cats' must be used") 
            triple_file = ".\\Dataset\\Dataset-Mturk\\triple_list.csv"
        
        if triple_file != None:
            self.dataset = CADDataset(root, file_name, force_reprocess=True, triple_file=triple_file)
            return
        if add_cats == True:
            self.dataset = CADDataset(root, file_name, force_reprocess=True, add_cats=add_cats)
            return
        
        assert (triple_file == None) and (add_cats == False), "At least one of 'triple_file' or 'add_cats' must be used"
    
    def info_dataset(self):
        print()
        print(f'Dataset: {self.dataset}:')
        print('====================')
        print(f'Number of graphs: {len(self.dataset)}')
        print(f'Number of node features: {self.dataset.num_node_features}')
        print(f'Number of edge features: {self.dataset.num_edge_features}')

        data, data1, data2 = self.dataset[0]  # Get the first graph object.

        print()
        print(data)
        print('=============================================================')

        # Gather some statistics about the first graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Number of edge features: {data.num_edge_features}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
        print(f'Contains self-loops: {data.contains_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        print(f'Number of node features: {data.num_node_features}')
    '''
    Training
    '''
    def train(self):
        
        print(self.args)
        assert self.dataset != None, "No dataset loaded"
        self.PGNN = PartGNN(self.dataset,self.args)
        self.PGNN.train()
        
    
    '''Useful functions'''

    def get_vectors(self, model_list):
        #### NOTE TO HUGH, I suspect this bit might change depending on how something is loaded from strembed
        ''' Takes a list of step files, converts them to correct format, then passes them through the network '''
        
        
        # uses step to graph2graph to convert a step into a gz file.
        tmp_file = "tmp/raw/temp2.gz"
        if not os.path.exists("tmp/raw/"):
            os.makedirs("tmp/raw/")
        
        create_dataset(model_list,tmp_file)
        
        #edit create_dataset, so "dataset folder" can be a model list 
        
        model_dataset = CADDataset("tmp",filename="temp2.gz", force_reprocess=True)
        # load model dataset
        
        
        #print("Models for vectors loaded")
        # print("model_dataset:", model_dataset)
        
        self.load_model(model_loc=self.args.model_loc)
        
        ''' Return vectors of list of models '''
        vector_array = self.PGNN.get_vectors(model_dataset)
        
        model_dict = {}
        
        for i, name in enumerate(model_dataset.filenamelist):
            print(i, name)
            model_dict[name[0]] = vector_array[i]
        return model_dict
        
    
    def compare_pairs(self,model_list):
        '''
        Load two models and compare them
        At the moment this just calls get_vectors and returns results from that
        
        '''
        model_dict = self.get_vectors(model_list)
        list = [model_dict]
        dist = np.linalg.norm(model_dict[list[0]]-model_dict[list[1]])
        print("distance",dist)
        return dist, model_dict
        
    def find_in_dataset(self,input_vector,dataset_array,list_length=None):
        '''
        Compare given model with dataset and return list
        '''
        dist_dict = {}
        for vector in dataset_array:
            dist = np.linalg.norm(vector_array[0]-vector_array[1])
            dist_list.append(dist)
        
        pass
        
        
    def predict_cat(self):
        pass
        
        
    
    def cat_model_init(self):
        
        pass
        

    def load_model(self,model_loc=None):
        '''
        Load model in advance of future tasks
        '''
        if self.model_loaded == False:
            
            if self.dataset == None:
                self.load_dataset()
            
            self.PGNN = PartGNN(self.dataset,self.args)
            self.PGNN.load_model()
            
            if model_loc != None:
                self.PGNN.load_model(model_loc=model_loc)
        else:
            pass
        


if __name__ == "__main__":
   print("Testing PartFind")
   pf = PartFind()
   modellist = ["./test_parts/TORCH BODY - UPPER.STEP.STEP.STEP.STEP.STEP"]
#   modellist = ["./test_parts/A.step","./test_parts/M6 HEX NUT_BS EN 24034 - M6 - C.STEP.STEP.STEP.STEP.STEP"]#"./test_parts/0000028089b.STEP"] # "./test_parts/0000026437.STEP"
   vectors = pf.get_vectors(modellist)
   print(vectors)
else:
   print("PartFind_v2 Imported")
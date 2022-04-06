# python class for partfind, load this then all the functionality should be in here.
# TH 03/2022


# partgnn isn't in this version, updating from deep cad graph
from CADDataset import CADDataset


class PartFind():
    def __init__(self):
        
        self.dataset_file = None
        self.dataset = None
        
        super(PartFind,self).__init__()
        
        
    '''
    Dataset management
    '''
        
        
    def load_dataset(self):
        self.dataset = CADDataset(".\\Dataset\\Dataset-Cakebox", "cakebox_nx.pickle", force_reprocess=False)
        
    def create_dataset(self,root='.\\Dataset',file_name= "ABC_nx.pickle",triple_file=None,add_cats=False):
        '''
        This generates a dataset from .pickled set of nx graphs
        root: The folder your pickle is in and where you want your dataset stored.
        file_name: name of pickle
        force_reprocess: Re processed dataset
        triple_file: is a cvs file containing predetermined triples of file names, a base model, a similar model and a dissimilar model
        add_cats: if true, then category data is used to find similar or dissimilar models
        '''
        if (triple_file == None) and (add_cats == False):
            print("Atleast one of 'triple_file' or 'add_cats' must be used") 
            triple_file = ".\\Dataset\\triple_data.csv"
        
        if triple_file != None:
            self.dataset = CADDataset(root, file_name, force_reprocess=True, triple_file=triple_file)
            return
        if add_cats == True:
            self.dataset = CADDataset(root, file_name, force_reprocess=True, add_cats=add_cats)
            return
        
        assert (triple_file == None) and (add_cats == False), "Atleast one of 'triple_file' or 'add_cats' must be used"
        
        

if __name__ == "__main__":
   print("Testing PartFind")
   pf = PartFind().create_dataset()   
else:
   print("PartFind_v2 Imported")
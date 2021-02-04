""" PartGNN """
#https://github.com/benedekrozemberczki/SimGNN/blob/master/src/simgnn.py#L212


# tomorrow to do
# add test for test_Dataset
# add batching


import torch
import dgl
import os
from Dataset.ABCSiameseDataset import ABCSiameseDataset
from Dataset.ABCDataset import ABCDataset
from utils import print_graph, compare_graphlets
from step2image_pyocc import render_step
from Model.model import PartGNN as GNN
from tqdm import tqdm
import math
import numpy as np


class PartGNN(torch.nn.Module):

    def __init__(self, args):
        """
        :param args: Arguments object
        """
        super(PartGNN, self).__init__()
        self.args = args
        self.verbose = self.args.verbose
        self.save_frequency = 1
        self.save_folder = "./trained_models/"
        if self.verbose:
            print("Verbose enabled")
        
        if os.name == 'nt':
            self.model_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data"
        else:
            self.model_folder = "/nobackup/prctha/dgl/Dataset/gz"
        self.step_dataset = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print("Using device:",self.device)
        self.dataset_loaded = False
        
        #self.create_dataset()
        #self.load_dataset()
        #self.setup_layers()
        self.test()
    
    def create_dataset(self):
        print("Loading dataset from step files...")
        model_folder = self.model_folder
        self.step_dataset = ABCDataset(raw_dir='Dataset/')
        
    
    def load_dataset(self):
        # load dataset
        print("Initalising dataset...")
        raw_dir = self.args.dataset
        dataset_range = self.args.dataset_range
        print("dataset_range",dataset_range)
        if dataset_range==None:
            dataset_range=[0,1000000]
        # to impliment load dataset from abc folder
        single_dataset = []
        model_folder = self.model_folder
        self.dataset = ABCSiameseDataset(
        self.step_dataset,
        raw_dir=raw_dir,
        verbose=self.verbose,
        force_reload=False,
        continue_dataset=False)
        
        self.dataset_loaded = True
        
        # ## test
        # print("test point")
        # for i in range(6,7):
            # print("i:",i)
            # graphs, targets, filenames = self.dataset[i]
            # #print("graphs", graphs)
            # print("filename:",filenames)
            # #model_folder = self.model_folder
# #            for filename in filenames:
# #                model_file = self.__gz2step(os.path.join(model_folder,filename))
# #                image1 = render_step(model_file,remove_tmp=False)
            # print("targets",targets)
            # g1,g2,g3 = dgl.unbatch(graphs)
            # #print("g1",g1)
            # #print("g2",g2)
            # #print("g3",g3)
            # print_graph(g1,str(i)+"tmp/g1.png")
            # print_graph(g2,str(i)+"tmp/g2.png")
            # print_graph(g3,str(i)+"tmp/g3.png")
            # values2, _ = compare_graphlets([g1,g2,g3])
            # print("values2",values2)

    def __gz2step(self,filename):
        return os.path.splitext(filename)[0]+".step"
        
        
    def train(self):
        print("Initalising....")
        if not self.dataset_loaded:
            print("self.dataset_loaded",self.dataset_loaded)
            self.load_dataset()
        
        batch_size = self.args.batch_size
        num_epochs = self.args.epochs
        
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                    batch_size=batch_size, collate_fn=self.__collate_single)
                                                    
        
        model = GNN(self.args)
        
        
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        
        model.train()
        
        for epoch in range(0,self.args.epochs):
            print("Epoch",epoch+1, "...")
            
            self.loss_sum = 0
            main_index = 0
            
            
            for batch_idx, (data, targets, file_names) in enumerate(tqdm(train_loader)):
                #print("batch_idx",batch_idx)
                # print("data",data)
                # print("targets",targets)
                # print("file_names",file_names)
                self.optimizer.zero_grad()
                
                # unbatch
                gb = dgl.unbatch(data)
                # print("gb[0::3]",gb[0::3])
                # group into three batches
                batch_0 = dgl.batch(gb[0::3]).to(self.device)
                batch_1 = dgl.batch(gb[1::3]).to(self.device)
                batch_2 = dgl.batch(gb[2::3]).to(self.device)
                
                # missing step transfer to torch
                
                # print("batch_0.batch_size",batch_0.batch_size)
                score_match = model(batch_0,batch_1) # sample, match, not match
                score_unmatch = model(batch_0,batch_2)
                
                # print("score_match",score_match)
                
                if batch_size == 1:
                    targets_match = targets[0]
                    targets_match = targets_match.reshape(1,1)
                    targets_match = targets_match.type(torch.float32)
                    targets_unmatch = targets[1]
                    targets_unmatch = targets_unmatch.reshape(1,1)
                    targets_unmatch = targets_unmatch.type(torch.float32)
                else:
                    targets_match = torch.squeeze(targets[:,0])
                    targets_match = targets_match.type(torch.float32)
                    targets_unmatch = torch.squeeze(targets[:,1])
                    targets_unmatch = targets_unmatch.type(torch.float32)
                
                # print("targets_match.shape",targets_match.shape)
                # print("score_match.shape",score_match.shape)
                
                loss_match = torch.nn.functional.mse_loss(targets_match,score_match)
                loss_unmatch = torch.nn.functional.mse_loss(targets_unmatch,score_unmatch)
                losses = loss_match + loss_unmatch
                
                losses.backward() # retain_graph=True
                self.optimizer.step()
                loss_score = losses.item()
                
                self.loss_sum = self.loss_sum + loss_score * batch_size
                main_index = main_index + batch_size
                loss = self.loss_sum/main_index
                
            print("Epoch",epoch+1,"loss =",loss)
            if epoch & self.save_frequency == 0:
                torch.save(model, os.path.join(self.save_folder,'partfind_{:03}.pt').format(epoch))
                
    def __collate_single(self,batch):
        assert len(batch) == 1, 'Currently we do not support batched training'
        return batch[0]
        
    def __collate(self,samples):
        #currently not supported
        # print("samples:",samples)
        graphs, targets, file_names = map(list, zip(*samples))
        # print("graphs:",graphs)
        # print("graphslen",len(graphs))
        # print("file_names",file_names)
        batched_graph = dgl.batch(graphs)
        # print("targets:",targets)
        batched_targets = torch.stack(targets)
        # print("batched_targets:",batched_targets)
        return batched_graph, batched_targets , file_names
        

    def test(self):
        print("Testing....")
        if not self.dataset_loaded:
            print("self.dataset_loaded",self.dataset_loaded)
            self.load_dataset()
        
        batch_size = self.args.batch_size
        num_epochs = self.args.epochs
        
        test_loader = torch.utils.data.DataLoader(self.dataset,
                                                    batch_size=1, collate_fn=self.__collate_single)
        scores = []
        
        
        model = GNN(self.args)
        
        #chooses most recent file if none specified # needs implimenting
        def newest(path):
            files = os.listdir(path)
            paths = [os.path.join(path, basename) for basename in files]
            return max(paths, key=os.path.getctime)
        model_path = newest(self.save_folder)
        print("loading model:",model_path)
        model = torch.load(model_path)
        
        self.scores = []
        self.scores_match = []
        self.scores_unmatch = []
        self.ground_truth = []
        for batch_idx, (data, targets, file_names) in enumerate(tqdm(test_loader)):
            #batch should be 1 for this
            gb = dgl.unbatch(data)
                # print("gb[0::3]",gb[0::3])
                # group into three batches
            batch_0 = dgl.batch(gb[0::3]).to(self.device)
            batch_1 = dgl.batch(gb[1::3]).to(self.device)
            batch_2 = dgl.batch(gb[2::3]).to(self.device)
            
            #grouth truth
            gt_match = targets[0].numpy()
            gt_unmatch = targets[1].numpy()
            
            self.ground_truth.append(gt_match)
            self.ground_truth.append(gt_unmatch)
            
            
            prediction_match = model(batch_0,batch_1) # sample, match, not match
            prediction_unmatch = model(batch_0,batch_2)
            
            loss_match = self.calculate_loss(prediction_match,targets[0])
            loss_unmatch = self.calculate_loss(prediction_unmatch,targets[1])
            
            self.scores.append(loss_match)
            self.scores.append(loss_unmatch)
            
            print("Predicted",prediction_match,"actual",gt_match)
            print("Predicted",prediction_unmatch,"actual",gt_unmatch)
            
        self.print_evaluation()
    
    def calculate_loss(self,prediction, target):
        """
        Calculating the squared loss on the normalized GED.
        :param prediction: Predicted log value of GED.
        :param target: Factual log transofmed GED.
        :return score: Squared error.
        """
        prediction = -math.log(prediction)
        try:
            target = -math.log(target)
        except:
            target = -math.log(0.01)
        score = (prediction-target)**2
        return score
     
     
    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error, 5))+".")
        print("\nModel test error: " +str(round(model_error, 5))+".")
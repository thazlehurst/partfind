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
#from step2image_pyocc import render_step

from Model.model import PartGNN as GNN
from tqdm import tqdm
import math
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


import grakel
from grakel.utils import graph_from_networkx
from grakel.kernels import GraphletSampling
import networkx as nx

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
        #self.train()
        #self.test()
        #self.create_lib()
        #self.load_model_find()
        #self.find_parts([],n=3)
        
    
    def create_dataset(self):
    #Function to create the dataset
    
        print("Loading dataset from step files...")
        model_folder = self.model_folder
        self.step_dataset = ABCDataset(raw_dir='Dataset/')
        print("Step files loaded")
        print("Creating siamese dataset...")
        
        raw_dir = self.args.dataset
        dataset_range = self.args.dataset_range
        
        self.dataset = ABCSiameseDataset(
        self.step_dataset,
        raw_dir=raw_dir,
        verbose=self.verbose,
        dataset_range=dataset_range,
        force_reload=True,
        continue_dataset=True)
        
    
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
        
        
        
        batch_size = self.args.batch_size
        validation_split = .2
        shuffle_dataset = True
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(val_indices)
        
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
                                                    batch_size=batch_size, collate_fn=self.__collate_single,sampler=self.train_sampler)
                                                    
        
        model = GNN(self.args)
        model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        
        model.train()
        
        for epoch in range(0,self.args.epochs):
            print("Epoch",epoch+1, "...")
            
            self.loss_sum = 0
            main_index = 0
            
            skipped = 0
            
            for batch_idx, (data, targets, file_names) in enumerate(tqdm(train_loader)):
                #print("batch_idx",batch_idx)
                # print("data",data)
                # 
                #if batch_idx == 1:
                    #print("targets",targets)
                # print("file_names",file_names)
                
                if targets[0] != targets[0]:
                    #print("targets[0]",targets[0])
                    skipped += 1
                    continue
                elif targets[1] != targets[1]:
                    #print("targets[1]",targets[1])
                    skipped += 1
                    continue
                
                self.optimizer.zero_grad()
                
                # unbatch
                gb = dgl.unbatch(data)
                # print("gb[0::3]",gb[0::3])
                # group into three batches
                batch_0 = dgl.batch(gb[0::3]).to(self.device)
                batch_1 = dgl.batch(gb[1::3]).to(self.device)
                batch_2 = dgl.batch(gb[2::3]).to(self.device)
                
                targets = targets.to(self.device)
                
                
                
                
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
                
#                if loss_match != loss_match:
#                    print('batch_0',batch_0)
#                    print('batch_1',batch_1)
#                    print('batch_2',batch_2)
#                    print('targets',targets)
#                    print('targets_match',targets_match)
#                    print('loss_match',loss_match)
#                if loss_unmatch != loss_unmatch:
#                    print('batch_0',batch_0)
#                    print('batch_1',batch_1)
#                    print('batch_2',batch_2)
#                    print('targets',targets)
#                    print('targets_match',targets_unmatch)
#                    print('loss_unmatch',loss_unmatch)
                
                
                losses.backward() # retain_graph=True
                self.optimizer.step()
                loss_score = losses.item()
                
                self.loss_sum = self.loss_sum + loss_score * batch_size
                main_index = main_index + batch_size
                loss = self.loss_sum/main_index
            
            print("Skipped:",skipped)    
            print("Epoch",epoch+1,"loss =",loss)
            if True:#epoch & self.save_frequency == 0:
                torch.save(model, os.path.join(self.save_folder,'partfind_{:03}.pt').format(epoch))
            self.test()
                
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
                                                    batch_size=1, collate_fn=self.__collate_single,sampler=self.valid_sampler)
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
        model.eval()
        self.scores = []
        self.scores_match = []
        self.scores_unmatch = []
        self.ground_truth = []
        skipped = 0
        match_correct = 0
        unmatch_correct = 0
        compare_correct = 0
        total = 0
        for batch_idx, (data, targets, file_names) in enumerate(tqdm(test_loader)):
            #batch should be 1 for this
            if targets[0] != targets[0]:
                #print("targets[0]",targets[0])
                skipped += 1
                continue
            elif targets[1] != targets[1]:
                #print("targets[1]",targets[1])
                skipped += 1
                continue
            
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
            
            match_match = self.is_match(prediction_match,targets[0])
            match_unmatch = self.is_match(prediction_unmatch,targets[1])
            if match_match:
                match_correct += 1
            if match_unmatch:
                unmatch_correct += 1
                
            if prediction_match > prediction_unmatch:
                compare_correct += 1 
            
            self.scores.append(loss_match)
            self.scores.append(loss_unmatch)
            
            # print("Predicted",prediction_match,"actual",gt_match)
            # print("Predicted",prediction_unmatch,"actual",gt_unmatch)
            total += 2
            
        
        #print("skipped:",skipped)
        
        total_correct = match_correct+unmatch_correct
        
        correct_percent = ( total_correct/(total))
        match_percent = (match_correct/(total/2))
        unmatch_percent = (unmatch_correct/(total/2))
        
        percentage = "{:.0%}".format(correct_percent)
        match_percentage = "{:.0%}".format(match_percent)
        unmatch_percentage = "{:.0%}".format(unmatch_percent)
        
        print("Correct:",percentage)
        print("Match percent",match_percentage)
        print("Unmatch percent",unmatch_percentage)
        
        compare_percent = (compare_correct/(total/2))
        compare_percentage = "{:.0%}".format(compare_percent)
        print("Compare percent",compare_percentage) 
        
        self.print_evaluation()
    
    def calculate_loss(self,prediction, target):
        """
        Calculating the squared loss on the normalized GED.
        :param prediction: Predicted log value of GED.
        :param target: Factual log transofmed GED.
        :return score: Squared error.
        """
        try:
            prediction = -math.log(prediction)
        except:
            #print("prediction error:",prediction)
            prediction = -math.log(0.01)
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
        
    def is_match(self,prediction,target):
        dist = torch.dist(prediction,target)
        if dist < 0.1:
            return True
        else:
            return False
            
    def test_pair(self,g0,g1):
        # input two graphs, output = score
        with torch.no_grad():
            score = self.model_find(g0,g1)
        return score.numpy()[0][0]
            
    def create_lib(self):
        print("Create vector library....")
        
        graphs_dataset = ABCDataset(raw_dir='Dataset/')
        print("graphs_dataset",graphs_dataset)
        model = GNN(self.args)
        
        #chooses most recent file if none specified # needs implimenting
        def newest(path):
            files = os.listdir(path)
            paths = [os.path.join(path, basename) for basename in files]
            return max(paths, key=os.path.getctime)
            
        model_path = newest(self.save_folder)
        print("loading model:",model_path)
        model = torch.load(model_path)
        model.eval()
        
        for i, (graph, filename) in enumerate(graphs_dataset):
            f1, f2 =model(graph,graph,mode='vector')
            print("f1",i)
    
    def load_model(self,load_dataset=False) :
        self.model_find = GNN(self.args).to(self.device)
        
        #chooses most recent file if none specified # needs implimenting
        def newest(path):
            files = os.listdir(path)
            paths = [os.path.join(path, basename) for basename in files]
            return max(paths, key=os.path.getctime)
            
        model_path = newest(self.save_folder)
        print("loading model:",model_path)
        self.model_find = torch.load(model_path,map_location=self.device)
        self.model_find.eval()
        
        if load_dataset:
            print("loading dataset...")
            self.graphs_dataset = ABCDataset(raw_dir='Dataset/')
        
    def find_in_data(self,filename_tofind):
        print("finding...",filename_tofind)
        for i, (graph, filename) in enumerate(tqdm(self.graphs_dataset)):
            # if i == 1:
                # print("test",filename)
            if filename_tofind == filename:
                print("found!")
                return graph
                
                
    def find_parts(self,test_g, filename_g,n=3,test=False):
        #test_g is model to compare with
        if test_g == []:
            test_g, filename_g = self.graphs_dataset[11]
        scores = []
        files = []
        g_list = [test_g]
        with torch.no_grad():
            print("searching...")
            for i, (graph, filename) in enumerate(tqdm(self.graphs_dataset)):
                if filename_g != filename:
                    score = self.model_find(test_g,graph)
                    #print("score",score.numpy()[0][0])
                    sc_np = score.numpy()[0][0]
                    scores.append(sc_np)
                    files.append(filename)
                    g_list.append(graph)
                # if sc_np>0.6:
                    # print(sc_np,filename)
                if i > 500:
                    break
                
        # find top n
        #print("scores",scores)
        sorted = np.flip(np.argsort(scores))
        #print("sorted",sorted)
        #print("filename_g",filename_g)
        file_list = []
        score_list = []
        for i in range(0,n):
            file_list.append(files[sorted[i]])
            score_list.append(scores[sorted[i]])
            
        g_score = []
        if test==True:
            print("calculating graphlets...")
            gl_kernel = GraphletSampling(normalize=True)
            grakels = self.dgl_grakel(g_list)
            grak_list = []
            for i, gr in enumerate(grakels):
                grak_list.append(gr)
            gl_kernel.fit_transform([grak_list[0]])
            g_score = []
            for grak in grak_list[0:]:
                fit, _ = gl_kernel.transform([grak])
                g_score.append(fit)
        
        # print("filename top",files[sorted[0]],scores[sorted[0]])
        # print("filename 2nd",files[sorted[1]],scores[sorted[1]])
        # print("filename 3rd",files[sorted[2]],scores[sorted[2]])
        return file_list, score_list, g_score

    def dgl_grakel(self,g):
        # convert dgl graph to grakel graph
        nx_list = []
        for graph in g:
          # 1. dgl to networkx
          nx_graph = dgl.to_networkx(graph)
          # 2. networkx to grakel
          for node in nx_graph.nodes():
            nx_graph.nodes[node]['label'] = node
          nx_list.append(nx_graph)
            
        krakel_graphs = graph_from_networkx(nx_list,as_Graph=True,node_labels_tag='label')
        # print("grakel:",g)
        return krakel_graphs
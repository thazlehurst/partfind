# TAH 02/12/2020
""" Takes ABC_Dataset of face adjancy graphs
for each graph in model it finds similar and dissimilar graph
creates data set where each data has three graphs
"""

from dgl.data import DGLDataset
import grakel
from grakel.utils import graph_from_networkx
import random
from grakel.kernels import GraphletSampling

import networkx as nx
import dgl
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from dgl.data import DGLDataset
from dgl.nn.pytorch import GraphConv, HeteroGraphConv

import random
import grakel
from grakel.utils import graph_from_networkx
from grakel.kernels import GraphletSampling
from six import iteritems
import os
import numpy as np
from collections import defaultdict
import json

import dill as pickle
import torch

from time import time
from tqdm import tqdm
# super(ABCDataset, self).__init__(name='abc_dataset',
#                                      url=url,
#                                      raw_dir=raw_dir,
#                                     #  save_dir=save_dir,
#                                      force_reload=force_reload,
#                                      verbose=verbose)

class SiameseDataset(DGLDataset):
  def __init__(self,dataset,raw_dir='',force_reload=False,verbose=True,continue_dataset=False,dataset_range=[0,100000]):
    #self.dataset = dataset
    self.threshold = 0.5
    self.batches = []
    self.targets = []
    self.file_list = []
    self.continue_dataset = continue_dataset
    self.dataset_range = list(range(dataset_range[0],dataset_range[1]))
    self.test_size = 10000000
    self.unmatched_dataset = torch.utils.data.Subset(dataset,self.dataset_range)
    self.face_count = defaultdict(int)
    self.edge_count = defaultdict(int)
    super(SiameseDataset,self).__init__(name='abc_siem',raw_dir=raw_dir,force_reload=force_reload,verbose=True)
    
    # self.process()
  
  def process(self):
    print("processing dataset...")
    total_start = time()
    print("checking if grahlets already counted...")
    loaded = False
    if loaded: #os.path.isfile(os.path.join(self.raw_dir,'graphlets.pkl')):
      print("loading graphlets")
      with open(os.path.join(self.raw_dir,'graphlets.pkl'),'rb') as phi_in:
           phi_z = pickle.load(phi_in)
      with open(os.path.join(self.raw_dir,'kernel.pkl'),'rb') as gl_in:
        gl_kernel = pickle.load(gl_in)
      with open(os.path.join(self.raw_dir,'file_list_done.pkl'),'rb') as list_in:
        file_list_done = pickle.load(list_in)
      loaded = True
    else:
      gl_kernel = GraphletSampling(normalize=True)
      file_list_done = []
    
    if self.continue_dataset:
      self.load()
      print("Current datset size:",len(self.file_list))
      print("will add the following...")
      print("Processing files", self.dataset_range[0], " to ", self.dataset_range[-1])
    
    g_list = []
    for i, (graph, file_name) in enumerate(self.unmatched_dataset):
      if self.verbose and i % 10000 == 0:
        print("Processing file ", i)
      if file_name not in file_list_done:
        edges_number = graph.number_of_edges()
        face_number = graph.number_of_nodes()
        self.edge_count[edges_number] += 1
        self.face_count[face_number] += 1
        if edges_number < 500:
#          if i == 7777:
#            print("skipping",i,"with face_count and edge_count",face_number,edges_number)
#            pass
          g_list.append(graph)
          file_list_done.append(file_name)
        else:
          pass
          print("Skipped ",file_name," too many edges ",edges_number)
    print("converting to grakels...")
    grakels = self.dgl_grakel(g_list)
    print("creating grakel list...")
    grak_list = []
    for i, gr in enumerate(grakels):
      grak_list.append(gr)
    
    json1 = json.dumps(self.face_count)
    f = open("face_count.json","w")
    f.write(json1)
    f.close
    json2 = json.dumps(self.edge_count)
    f = open("edge_count.json","w")
    f.write(json2)
    f.close
    
    
    print("Number of graphs", len(file_list_done))
    #assert 1 == 0
    # initialise kernel
    if not loaded:
      print("Test i:",i)
      gl_kernel.fit_transform([grak_list[i]])
    print("finding graphets....")
    # parse grak_list to get subgraphs in bins as dict
    Z = dict()
    for i, grak in enumerate(grak_list):
      if i > self.test_size:
        print("reached i > 500 for testing")
        break ## for testing
      if self.verbose:
        if i % 1 == 0:
          print("Processing file ", i)
      fit, z_row = gl_kernel.transform([grak])  # this has been modified in the module to produce graphlet bins also
      for ((k, j), v) in iteritems(z_row):
        Z[(i,j)] = v

    def get_bins(X_g):
      listx = []
      listy = []
      for gs in X_g:
        keys = X_g.keys()
        for (i,j) in keys:
          if i not in listx:
            listx.append(i)
          if j not in listy:
            listy.append(j)
      #print("listx",listx)
      #print("listy",listy)
      return max(listx)+1, max(listy)+1
    #print("Z.keys",Z.keys())
    lenx, leny = get_bins(Z)
    phi_z = np.zeros(shape=(lenx,leny)) ##
    #print("phi_z.shape",phi_z.shape)

    for ((i, j), v) in iteritems(Z):
      #print("i,j:",i,j," v:",v)
      phi_z[i, j] = v
    
    def save_step(phi_z,gl_kernel,file_list_done):
      print("saving graphlets....")
      with open(os.path.join(self.raw_dir,'graphlets.pkl'),'wb') as output:
        pickle.dump(phi_z,output,pickle.HIGHEST_PROTOCOL)
      with open(os.path.join(self.raw_dir,'kernel.pkl'),'wb') as output:
        pickle.dump(gl_kernel,output,pickle.HIGHEST_PROTOCOL)
      with open(os.path.join(self.raw_dir,'file_list_done.pkl'),'wb') as output:
        pickle.dump(file_list_done,output,pickle.HIGHEST_PROTOCOL)
      print("graphlets saved")
     
    #save_step(phi_z,gl_kernel,file_list_done)
    

    # matching part
    def isNaN(num):
        return num != num
    
    pbar = tqdm(total=len(file_list_done))
    bar_update = 1
    
    py_phi_z = torch.from_numpy(phi_z)
    
    for i, graph in enumerate(g_list):
      file_name = file_list_done[i]
      if i > self.test_size:
        print("reached i > 50 for testing")
        break ## for testing
      if i % bar_update == 0:
        pbar.update(bar_update)
      graph_0 = graph
      phi_x = np.expand_dims(phi_z[i],axis=0)
      """
      print("phi_x.shape",phi_x.shape)
      print("phi_x.T.shape",phi_x.T.shape)
      
      km = np.dot(phi_z, phi_x.T) #phi_z[:, :len(gl_kernel._graph_bins)]
      X_diag = np.sum(np.square(phi_x),axis=1)
      Z_diag = np.sum(np.square(phi_z),axis=1)
      km /= np.sqrt(np.outer(Z_diag, X_diag))
      print("km",km)
      closest = sorted(range(len(km)), key=lambda k: km[k])
      found_closest = False
      j = 0

      while found_closest != True:
        j -= 1
        print("j",j,"closest[j]",km[closest[j]])
        if isNaN(km[closest[j]]) == False:
          found_closest = True
          similar = closest[j]
      
      dissimilar = closest[0]
      print("similar",similar)
      print(km[similar])
      print("dissimilar",dissimilar)
      print(km[dissimilar])
      """
      # torch
      py_phi_x = torch.from_numpy(phi_x)
      #print("py_phi_x.shape",py_phi_x.shape)
      t = py_phi_x.T
      #print("py_phi_x.T.shape",py_phi_x.T.shape)
      #print("py_phi_z.shape",py_phi_z.shape)
      py_km = torch.mm(py_phi_z, py_phi_x.T) #py_phi_z[:, :len(gl_kernel._graph_bins)]
      py_X_diag = torch.sum(torch.square(py_phi_x),axis=1)
      py_Z_diag = torch.sum(torch.square(py_phi_z),axis=1)
      py_km /= torch.sqrt(torch.outer(py_Z_diag, py_X_diag))
      #print("py_km",py_km)
      
      closest = sorted(range(len(py_km)), key=lambda k: py_km[k])
      found_closest = False
      j = 0
      
      while found_closest != True:
        j -= 1
        #print("j",j,"closest[j]",py_km[closest[j]])
        try:
          if isNaN(py_km[closest[j]]) == False:
            found_closest = True
            similar = closest[j]
            valid = True
        except:
          found_closest = True
          valid = False
      # -1 in same
      dissimilar = closest[0]
      
      """
      print("similar",similar)
      print(km[similar])
      print("dissimilar",dissimilar)
      print(km[dissimilar])
      """
      
      
      if valid:
        graph_1, _ = self.unmatched_dataset[similar]
        graph_2, _ = self.unmatched_dataset[dissimilar]
        #target = torch.tensor([km[similar],km[dissimilar]])
        target = torch.tensor([py_km[similar],py_km[dissimilar]])
        self.batches.append(dgl.batch([graph_0,graph_1,graph_2]))
        self.targets.append(target)
        self.file_list.append(file_name)
        self.N = len(self.file_list)
    
    pbar.close()
    
    total_time = time() - total_start
    print("This took",total_time)

    #   print("graph_0 has ", graph_0.number_of_nodes(), " nodes and ", graph_0.number_of_edges(), " edges")
    #   grakels = self.dgl_grakel([graph_0])
    #   for graks in grakels:
    #     grakel_0 = [graks]

    #   fit = gl_kernel.fit_transform(grakel_0)
    #   print("fit:",fit)
    #   if fit != fit:
    #     print("nan")
    #     continue
    #   ti = 0
    #   tr_list = random.sample(range(len(dataset)-1),len(dataset)-1)

    #   while True:
    #     # find random graph, check its 
        
    #     index_1 = tr_list[ti]
    #     graph_1, _ = self.dataset[index_1]
    #     print("graph_1 has ", graph_1.number_of_nodes(), " nodes and ", graph_1.number_of_edges(), " edges")
    #     grakels = self.dgl_grakel([graph_1])
    #     for graks in grakels:
    #       grakel_1 = [graks]
    #     if self.verbose:
    #       print("calc sim")
    #     similarity_1 = gl_kernel.transform(grakel_1)
    #     print("similarity_1",similarity_1)
    #     print("true?",similarity_1 > self.threshold)
    #     if similarity_1 > self.threshold:
    #       break
    #     ti += 1
    #     if ti == len(dataset)-1:  # maybe change to max similarity
    #       graph_1, _ = self.dataset[i]
    #       break

    #   while True:
    #     # find random graph, check its 
    #     index_2 = random.randint(0, len(dataset)-1)
    #     graph_2, _ = self.dataset[index_2]
    #     print("graph_2 has ", graph_2.number_of_nodes(), " nodes and ", graph_2.number_of_edges(), " edges")
    #     grakels = self.dgl_grakel([graph_2])
    #     for graks in grakels:
    #       grakel_2 = [graks]
    #     similarity_2 = gl_kernel.transform(grakel_2)
    #     if similarity_2 < (1-self.threshold):
    #       break

    #   target = th.tensor([similarity_1,similarity_2])
    #   # orginal, close, far
    #   # return graph_0, graph_1, graph_2
    #   # batch graphs instead
    #   self.batches.append(dgl.batch([graph_0,graph_1,graph_2]))
    #   self.targets.append(target)
    # self.N = i
    print("datset processed")

  def __getitem__(self,index):    
    
    return self.batches[index], self.targets[index]


  def __len__(self):
    return len(self.batches)

  def save(self):
    # save processed data to directory `self.save_path`
    print("saving dataset...")
    graph_path = os.path.join(self.save_path,'abc_siam_{}.bin'.format(self.name))
    info_path = os.path.join(self.save_path,'abc_siam_{}.pkl'.format(self.name))
    info_dict = {'N': self.N,
                 'targets': self.targets, 'file_list': self.file_list}
    graph_export = []
    for batch in self.batches:
      g0,g1,g2 = dgl.unbatch(batch)
      graph_export.append(g0)
      graph_export.append(g1)
      graph_export.append(g2)
    save_graphs(str(graph_path),graph_export,{})
    save_info(str(info_path),info_dict)
    print("dataset saved!")
    
  def load(self):
    print("loading...")
    # load processed data from directory `self.save_path`
    graph_path = os.path.join(self.save_path,'abc_siam_{}.bin'.format(self.name))
    info_path = os.path.join(self.save_path,'abc_siam_{}.pkl'.format(self.name))
    print("loading dataset from file...")
    
    info_dict = load_info(str(info_path))

    
    self.N = info_dict['N']
    self.targets = info_dict['targets']
    self.file_list = info_dict['file_list']
    graph_import, _ = load_graphs(str(graph_path))
    # print("lengraph_import",len(graph_import))
    for i in range(0,len(graph_import),3):
      self.batches.append(dgl.batch([graph_import[i],graph_import[i+1],graph_import[i+2]]))
    # self.batches = batches[0]

  def has_cache(self):
    print("checking...")
    # check whether there are processed data in `self.save_path`
    graph_path = os.path.join(self.save_path,'abc_siam_{}.bin'.format(self.name))
    info_path = os.path.join(self.save_path,'abc_siam_{}.pkl'.format(self.name))
    if os.path.exists(graph_path) and os.path.exists(info_path):
      print("file exists")
      return True
    print("file doesn't exist")
    return False
  
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
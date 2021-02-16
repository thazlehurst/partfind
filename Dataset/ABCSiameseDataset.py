"""
class for loading Siamese part dataset, created from ABC dataset
"""
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import torch
from collections import defaultdict
import random
import grakel
from grakel.utils import graph_from_networkx
from grakel.kernels import GraphletSampling
from six import iteritems
import pickle
import os
from time import time
from tqdm import tqdm
import numpy as np

class ABCSiameseDataset(DGLDataset):
  def __init__(self,dataset,raw_dir='',force_reload=False,verbose=True,continue_dataset=False,dataset_range=[0,100000]):
    self.dataset = dataset
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
    super(ABCSiameseDataset,self).__init__(name='abc_siem',raw_dir=raw_dir,force_reload=force_reload,verbose=True)
    
    # self.process()
  
  def process(self):
    print("processing dataset...")
    total_start = time()
    print("checking if grahlets already counted...")
    loaded = False
    
    ### currently doesn't use this  ## doesnt need to if using large batch number
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
    
    if self.continue_dataset:  ## load previously processed data
      self.load()
      print("Current datset size:",len(self.file_list))
      print("will add the following...")
      print("Processing files", self.dataset_range[0], " to ", self.dataset_range[-1])
    
    g_list = []
    unmatched_dataset_filtered = []
    for i, (graph, file_name) in enumerate(self.unmatched_dataset):
      if self.verbose and i % 10000 == 0:
        print("Processing file ", i)
      if file_name not in file_list_done:
        edges_number = graph.number_of_edges()
        face_number = graph.number_of_nodes()
        self.edge_count[edges_number] += 1
        self.face_count[face_number] += 1
        if edges_number < 500:
          g_list.append(graph)
          file_list_done.append(file_name)
          unmatched_dataset_filtered.append(self.unmatched_dataset[i])
          #print("Passed ",file_name,"edges ",edges_number)
        else:
          pass
          #print("Skipped ",file_name," too many edges ",edges_number)
    # print("There are",len(g_list),"valid graphs")
    # #print("unmatched_dataset_filtered",unmatched_dataset_filtered)
    # graph_t, filenames_t = unmatched_dataset_filtered[0]
    # print("graph_t",graph_t)
    # print("filenames_t",filenames_t)
    
    # print("converting to grakels...")
    grakels = self.dgl_grakel(g_list)
    print("creating grakel list...")
    grak_list = []
    for i, gr in enumerate(grakels):
      grak_list.append(gr)
    
    # json1 = json.dumps(self.face_count)
    # f = open("face_count.json","w")
    # f.write(json1)
    # f.close
    # json2 = json.dumps(self.edge_count)
    # f = open("edge_count.json","w")
    # f.write(json2)
    # f.close
    
    
    print("Number of graphs", len(file_list_done))
    #assert 1 == 0
    # initialise kernel
    if not loaded:  # loaded=false always
      print("Test i:",i)
#      print("grak_list[i]",grak_list[i])
      gl_kernel.fit_transform([grak_list[i]])
    print("finding graphets....")
    # parse grak_list to get subgraphs in bins as dict
    Z = dict()
    for i, grak in enumerate(grak_list):
      #print("i:",i,"grak:",grak.nv())
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
        graph_1, file_name_1 = unmatched_dataset_filtered[similar]
        graph_2, file_name_2 = unmatched_dataset_filtered[dissimilar]
        # print("similar",similar)
        # print("dissimilar",dissimilar)
        #target = torch.tensor([km[similar],km[dissimilar]])
        target = torch.tensor([py_km[similar],py_km[dissimilar]])
        self.batches.append(dgl.batch([graph_0,graph_1,graph_2]))
        self.targets.append(target)
        file_list = [file_name,file_name_1,file_name_2]
        # print("file_list",file_list)
        self.file_list.append(file_list)  #file_name
        self.N = len(self.file_list)
    
    pbar.close()
    
    total_time = time() - total_start
    print("This took",total_time)
    print("datset processed")

  def __getitem__(self,index):    
    
    return self.batches[index], self.targets[index], self.file_list[index]

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
    print("Dataset saved to ",self.save_path)

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
    n_models = len(graph_import)
    if self.verbose:
      pbar = tqdm(total=int(n_models/3))
    for i in range(0,n_models,3):
      self.batches.append(dgl.batch([graph_import[i],graph_import[i+1],graph_import[i+2]]))
      if self.verbose:
        pbar.update(1)
    if self.verbose:
      pbar.close()
    print("Previous dataset loaded")

  def has_cache(self):
    print("checking...")
    # check whether there are processed data in `self.save_path`
    if self.verbose:
        print("Looking for existing datset in ",self.save_path)
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
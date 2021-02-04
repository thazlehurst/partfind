import dgl
from dgl.data import DGLDataset
import networkx as nx
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

class ABCDataset(DGLDataset):
  """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
  """
  def __init__(self,
               url=None,
               raw_dir=None,
              #  save_dir=None,
               force_reload=False,
               verbose=False):
    
    self.graphs = []
    self.file_list = []
    # if save_dir != None:
    #   self._save_dir = save_dir

    super(ABCDataset, self).__init__(name='abc_dataset',
                                     url=url,
                                     raw_dir=raw_dir,
                                    #  save_dir=save_dir,
                                     force_reload=force_reload,
                                     verbose=verbose)

  def download(self):
    # download raw data to local disk
    # download from ABC website, make it so one can choose how much of the data is downloaded, but it also needs processing!
    #raise NotImplementedError()
    pass

  def process(self):
    # process raw data to graphs, labels, splitting masks
    if self.verbose:
      print('loading data...')
    if os.name == 'nt':
      root = "C:\\Users\\prctha\\PythonDev\\ABC_Out" #self.raw_path
    else:
      root = "/nobackup/prctha/dgl/Dataset/gz"
    i = 0
    for file in os.listdir(root):
      if file.endswith(".gz"):
        i += 1
        g_load = nx.read_gpickle(os.path.join(root,file))
        g_load = self.networkx_to_dgl(g_load)
        face_g = g_load.node_type_subgraph(['face'])
        g_out = dgl.to_homogeneous(face_g)
        if g_out.num_nodes() != 0:
          self.graphs.append(g_out)
          self.file_list.append(file)
      if i % 100 == 0:
        print('loaded ', i, ' files')
    self.N = i
    if self.verbose:
      print('dataset loaded')

  def __getitem__(self, idx):
    # get one example by index
    # returns dgl.DGLGraph, #and file name for lookup later
    return self.graphs[idx], self.file_list[idx]

  def __len__(self):
        # number of data examples
    return len(self.graphs)

  def save(self):
    # save processed data to directory `self.save_path`
    graph_path = os.path.join(self.save_path,'abc_{}.bin'.format(self.name))
    info_path = os.path.join(self.save_path,'abc_{}.pkl'.format(self.name))
    info_dict = {'N': self.N,
                 'file_list': self.file_list}
    save_graphs(str(graph_path),self.graphs,{})
    save_info(str(info_path),info_dict)

  def load(self):
    # load processed data from directory `self.save_path`
    
    graph_path = os.path.join(self.save_path,'abc_{}.bin'.format(self.name))
    info_path = os.path.join(self.save_path,'abc_{}.pkl'.format(self.name))
    if self.verbose:
      print("loading dataset from file...")
    
    info_dict = load_info(str(info_path))

    
    self.N = info_dict['N']
    self.file_list = info_dict['file_list']
    graphs = load_graphs(str(graph_path))
    self.graphs = graphs[0]
    print("Loaded dataset")

  def has_cache(self):
    # check whether there are processed data in `self.save_path`
    print("Attempting to load files from", self.save_path)
    graph_path = os.path.join(self.save_path,'abc_{}.bin'.format(self.name))
    info_path = os.path.join(self.save_path,'abc_{}.pkl'.format(self.name))
    if os.path.exists(graph_path) and os.path.exists(info_path):
      return True
    return False



  def networkx_to_dgl(self,A_nx):
    # need to convert it into something dgl can work with
    node_dict = {} # to convert from A_nx nodes to dgl nodes
    part_count = 0
    assembly_count = 0
    face_count = 0

    face_str = []
    face_dst = []
    link_str = []
    link_dst = []
    assembly1_str = []
    assembly1_dst = []
    assembly2_str = []
    assembly2_dst = []

    for node_str, node_dst, key, data in A_nx.edges(data=True,keys=True):
      t = data['type']
      # get nodes in dict
      tn_str = A_nx.nodes[node_str]['type']
      if node_str not in node_dict:
        if tn_str == 'part':
          node_dict[node_str] = part_count
          part_count += 1
        elif tn_str == "assembly":
          node_dict[node_str] = assembly_count
          assembly_count += 1
        elif tn_str == "face":
          node_dict[node_str] = face_count
          face_count += 1
      
      tn_dst = A_nx.nodes[node_dst]['type']
      if node_dst not in node_dict:
        if tn_dst == 'part':
          node_dict[node_dst] = part_count
          part_count += 1
        elif tn_dst == "assembly":
          node_dict[node_dst] = assembly_count
          assembly_count += 1
        elif tn_dst == "face":
          node_dict[node_dst] = face_count
          face_count += 1
      # there are three edge types so sort which ever one we are dealing with into that one

      if t == "face":
        assert tn_str == "face"
        assert tn_dst == "face"
        face_str.append(node_dict[node_str])
        face_dst.append(node_dict[node_dst])
      elif t == "link":
        # print("node_str, node_dst, key, data")
        # print(node_str, node_dst, key, data)
        # print("tn_str: ",tn_str)
        # print("tn_dst: ",tn_dst)
        assert tn_str == "face"
        assert tn_dst == "part"
        link_str.append(node_dict[node_str])
        link_dst.append(node_dict[node_dst])
      elif t == "assembly":
        assert tn_str == "assembly"
        assert tn_dst in ["assembly","part"]
        if tn_dst == "assembly":
          assembly1_str.append(node_dict[node_str])
          assembly1_dst.append(node_dict[node_dst])
        elif tn_dst == "part":
          assembly2_str.append(node_dict[node_str])
          assembly2_dst.append(node_dict[node_dst])

    # make heterograph
    A_dgl = dgl.heterograph({
      ('face','face','face') : ( face_str, face_dst ),
      ('face','link','part') : ( link_str, link_dst ), # part -> face
      ('assembly','assembly','part') : ( assembly2_str, assembly2_dst ), # these may be swapped around at some point
      ('assembly','assembly','assembly') : ( assembly1_str, assembly1_dst ),
      ('assembly','layer','part') : ([],[]), 
      ('part','layer','part') : ([],[]),
      ('assembly','layer','assembly') : ([],[]),
      ('part','layer','assembly') : ([],[])
    })

    return A_dgl
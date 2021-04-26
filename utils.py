import matplotlib.pyplot as plt
import networkx as nx
import dgl


import grakel
from grakel.utils import graph_from_networkx
from grakel.kernels import GraphletSampling

def compare_graphlets(g_list):
    gl_kernel = GraphletSampling(normalize=True,sampling={'n_samples': 10000})
    # gl_kernel = GraphletSampling(normalize=True)
    grakels = dgl_grakel(g_list)
    grak_list = []
    for i, gr in enumerate(grakels):
      grak_list.append(gr)
    # initialise kernel
    gl_kernel.fit_transform([grak_list[0]])
    matching = gl_kernel.transform(grak_list)
    # matching = []
    # for grak in grak_list[1:]:
      # matching.append(gl_kernel.transform([grak]))
    return matching

def print_graph(g,filename="graph.png"):
  #  save graph to image file,
  plot_size = (5,5)
  plt.figure(3,figsize=plot_size)
  # if g.device == 'cuda':
  g=g.cpu()
  if g.is_homogeneous:
    nx.draw(dgl.to_networkx(g))
  else:
    nx.draw(dgl.to_networkx(dgl.to_homogeneous(g)))
  #plt.show()
  plt.savefig(filename)
  plt.close()
  
def dgl_grakel(g):
    # convert dgl graph to grakel graph
    nx_list = []
    for graph in g:
      # 1. dgl to networkx
      graph=graph.cpu()
      nx_graph = dgl.to_networkx(graph)
      # 2. networkx to grakel
      for node in nx_graph.nodes():
        nx_graph.nodes[node]['label'] = node
      nx_list.append(nx_graph)
        
    krakel_graphs = graph_from_networkx(nx_list,as_Graph=True,node_labels_tag='label')
    # print("grakel:",g)
    return krakel_graphs
    

def graphlet_pair_compare(graph_1,graph_2):
    graph_list = [graph_1,graph_2]
    #print("converting graphs..")
    match_score = compare_graphlets(graph_list)
    #print("match_score",match_score[0])
    return match_score[1][0]
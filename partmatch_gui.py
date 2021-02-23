# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:41:06 2021

@author: prctha
"""



import streamlit as st

import numpy as np
import pandas as pd
import os, random
#import StringIO
import cv2
import networkx as nx
import dgl
import pickle
from step_to_graph import load_step



try:
	import OCC
	#st.write("OCC loaded")
except:
	st.write("OCC not loaded")
from step2image_pyocc import render_step

from partgnn import PartGNN
from main import parameter_parser

args = parameter_parser()
model = PartGNN(args)
model.load_model()

#from step2image import makeSnapshotWithoutGui

def load_gz(filepath):
    g_load = nx.read_gpickle(filepath)
    g_load = networkx_to_dgl(g_load)
    face_g = g_load.node_type_subgraph(['face'])
    g_out = dgl.to_homogeneous(face_g)
    return g_out
    
def load_from_step(filepath):
    s_load = load_step(filepath)
    g_load = networkx_to_dgl(s_load)
    face_g = g_load.node_type_subgraph(['face'])
    g_out = dgl.to_homogeneous(face_g)
    return g_out
    
def networkx_to_dgl(A_nx):
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
    
    #print("edges",A_nx.edges(data=True,keys=True))

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


#image_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Image"
model_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data"
tmp_folder = 'C:\\Users\\prctha\\PythonDev\\partfind\\tmp\\'
gz_folder = "C:/Users/prctha/PythonDev/ABC_Out/"
st.title('Part finder')

import sys


st.write("Upload part to find similar:")
uploaded_file = st.file_uploader("Choose first file")
uploaded_file2 = st.file_uploader("Choose second file")

if (uploaded_file is not None) and (uploaded_file2 is not None):
  # find image

  gz_file0 = os.path.splitext(uploaded_file.name)[0]
  gz_file0 = gz_file0 + ".gz"

  #graph0 = model.find_in_data(gz_file0)
  st.spinner()
  with st.spinner(text="Rendering file..."):
    image0 = render_step(uploaded_file)
    st.image(image0,caption="Input model",width=400)
  
  gz_file2 = os.path.splitext(uploaded_file2.name)[0]
  gz_file2 = gz_file2 + ".gz"
  
  st.spinner()
  with st.spinner(text="Rendering file..."):
    image2 = render_step(uploaded_file2)
    st.image(image2,caption="Input model",width=400)
    
  # st.write(uploaded_file)
  # gg0 = load_gz(os.path.join(gz_folder,gz_file0))
  G0 = load_from_step(uploaded_file)
  
  # gg2 = load_gz(os.path.join(gz_folder,gz_file2))
  G2 = load_from_step(uploaded_file2)
  
  
  # score = model.test_pair(gg0,gg2)
  # print("score",score)
  # st.write("score:",score)
  
  score = model.test_pair(G0,G2)
  print("score",score)
  st.write("score:",score)
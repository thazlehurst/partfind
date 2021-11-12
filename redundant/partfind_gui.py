# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:41:06 2021

@author: prctha
"""

# streamlit test


##import sys
##sys.path.append('C:\\Users\\prctha\\AppData\\Local\\Continuum\\anaconda3\\envs\\fc\\Library\\bin')

##import FreeCAD

import streamlit as st

import numpy as np
import pandas as pd
import os, random
#import StringIO
import cv2
import networkx as nx
import dgl
import pickle



try:
	import OCC
	st.write("OCC loaded")
except:
	st.write("OCC not loaded")
from step2image_pyocc import render_step

from partgnn import PartGNN
from main import parameter_parser

args = parameter_parser()
model = PartGNN(args)
model.load_model_find()

#from step2image import makeSnapshotWithoutGui

def load_gz(filepath):
    print("filepath",filepath)
    g_load = nx.read_gpickle(filepath)
    print("g_load",g_load)
    g_load = networkx_to_dgl(g_load)
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
    


#image_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Image"
model_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data"
tmp_folder = 'C:\\Users\\prctha\\PythonDev\\partfind\\tmp\\'
gz_folder = "C:/Users/prctha/PythonDev/ABC_Out/"
st.title('Part finder')

import sys


st.write("Upload part to find similar:")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  # find image
  #image_file0 = os.path.splitext(uploaded_file.name)[0]
  #image_file0 = os.path.join(image_folder,image_file0) + ".png"
  
  gz_file0 = os.path.splitext(uploaded_file.name)[0]
  gz_file0 = gz_file0 + ".gz"
  #gz_file0 = os.path.join(gz_folder,gz_file0) 
  #st.write("gzfile:",gz_file0)
  # print("gzfile:",gz_file0,os.path.isfile(gz_file0))
  # g_load = nx.read_gpickle(gz_file0)
  # print("g_load",g_load)
  graph0 = model.find_in_data(gz_file0)
  st.spinner()
  with st.spinner(text="Rendering file..."):
    image0 = render_step(uploaded_file)
    st.image(image0,caption="Input model",width=400)
  
  
  
  st.write("Finding similar models:")
  
  # # at the moment pick 3 files in folder at random
  with st.spinner(text="Finding results..."):
    size = 100000000
    #graph0 = load_gz(gz_file0)
    
    filelist, scorelist, g_score = model.find_parts(graph0,gz_file0,n=3,test=False)
    
    
    # while size > 500000:
      # size = 0
      # model_file1 = random.choice(os.listdir(model_folder)) #change dir name to whatever
      # model_file2 = random.choice(os.listdir(model_folder)) #change dir name to whatever
      # model_file3 = random.choice(os.listdir(model_folder)) #change dir name to whatever
      # model_file1 = os.path.join(model_folder,model_file1)
      # model_file2 = os.path.join(model_folder,model_file2)
      # model_file3 = os.path.join(model_folder,model_file3)
      # size = os.stat(model_file1).st_size
      # #st.write(os.stat(model_file1).st_size)
      # #st.write(os.stat(model_file2).st_size)
      # size = max(os.stat(model_file2).st_size,size)
      # #st.write(os.stat(model_file3).st_size)
      # size = max(os.stat(model_file3).st_size,size)
    
    def gz_to_step(file):
        file = os.path.splitext(file)[0]
        file = file + ".step"
        file = os.path.join(model_folder,file)
        return file
    
    print("g_score",g_score)
    #st.write("g_score",g_score)
    image1 = render_step(gz_to_step(filelist[0]))
    image2 = render_step(gz_to_step(filelist[1]))
    image3 = render_step(gz_to_step(filelist[2]))
    images = [image1,image2,image3]
    
    for i in range(0,3): # for 3 rows
      cols = st.beta_columns(2) # 2 columns
      cols[0].image(images[i],width=300)
      result_str = "Match: " + str(scorelist[i])# + " graphlet score: " + str
      cols[1].write(result_str)
  #st.image(images,caption=["Found model 1","Found model 2","Found model 3"],width= 300)
  

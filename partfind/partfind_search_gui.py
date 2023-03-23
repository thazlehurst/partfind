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
from step_to_graph import load_step
import sys



try:
    import OCC
    #st.write("OCC loaded")
except:
    st.write("OCC not loaded")
from step2image_pyocc import render_step, render_step_simple

from partgnn import PartGNN
from main import parameter_parser

#from step2image import makeSnapshotWithoutGui

def load_gz(filepath):
    g_load = nx.read_gpickle(filepath)
    g_load = networkx_to_dgl(g_load)
    face_g = g_load.node_type_subgraph(['face'])
    g_out = dgl.to_homogeneous(face_g)
    return g_out

def load_from_step(filepath):
    print("loading:",filepath)
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



def run_it():

    args = parameter_parser()
    model = PartGNN(args)
    model.load_model()

    model_folder = "C:\_Work\_DCS project\__ALL CODE\_Repos\StrEmbed-5-6\StrEmbed-5-6 for git\cakestep"
    model_folder = "C:\_Work\_DCS project\__ALL CODE\_Repos\StrEmbed-5-6\StrEmbed-5-6 for git\Torch Assembly"
    #model_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data" #change this folder to where your step files are saved

    cwd = os.getcwd()
    save_folder = os.path.join(cwd,"save_data")

    # Switch for list or image mode; change manually here for now
    list_mode = True



    st.title('Part finder')
    st.write("Upload part to find similar:")

    uploaded_file = st.file_uploader("Choose a file")

    # Return if no file
    if uploaded_file is None:
        print('File not found; returning')
        return

    # find image
    csv_loc = os.path.join(save_folder,uploaded_file.name) +".csv"

    load_csv=False
    try:
        prev_scores = pd.read_csv(csv_loc).set_index('file')
        scores_dict = prev_scores.to_dict('index')
        print("scores_dict:",scores_dict)
        load_csv=True
        print("loading from saved results")
    except:
        pass

    st.spinner()

    with st.spinner(text="Rendering file..."):
        image0 = render_step(uploaded_file)
        # image0 = render_step(uploaded_file, remove_tmp = False)
        st.image(image0,caption="Input model",width=400)
        # st.image(image0,caption="Input model",width=200)

    st.write("Finding similar models:")



    # # at the moment pick 3 files in folder at random
    with st.spinner(text="Finding results..."):

        def gz_to_step(file):
            file = os.path.splitext(file)[0]
            file = file + ".step"
            file = os.path.join(model_folder,file)
            return file
        i = 0

        main_graph = load_from_step(uploaded_file)

        file_list = []
        score_list = []
        for filename in os.listdir(model_folder):
            # i += 1
            # if i > 10:
                # break
            if filename.endswith(".step") or filename.endswith(".stp") or filename.endswith(".STP") or filename.endswith(".STEP"):

                try:
                    score = scores_dict[filename]['score']
                except:
                    check_graph = load_from_step(os.path.join(model_folder, filename))
                    score = model.test_pair(main_graph,check_graph)
                file_list.append(filename)
                score_list.append(score)
            else:
                continue


        # List mode (TH original version)
        if list_mode:
            df = pd.DataFrame(
            score_list, index=file_list, columns=['score'])
            df.index.name='file'

            df.sort_values(by=['score'], inplace=True, ascending=False)

            print("df:",df)

            df.to_csv(csv_loc)

            st.dataframe(df)  # Same as st.write(df)



        # Image mode (HR version)
        else:
            # # Open renderer to avoid new renderer being opened for each part image
            # renderer = Viewer3d()
            # renderer.Create()
            # renderer.SetSize(600,600)
            # renderer.SetModeShaded()

            file_dict = dict(sorted(zip(score_list, file_list), reverse = True))

            c_head = st.beta_columns(3)
            c_head[0].write('Shape image')
            c_head[1].write('Shape name')
            c_head[2].write('Similarity score')

            for k,v in file_dict.items():

                filename = v
                score = k

                # Get/create image
                im_name = os.path.splitext(filename)[0] + ".png"
                im_name = os.path.join(model_folder, im_name)
                if os.path.isfile(im_name):
                    # st.write('Found image')
                    im = cv2.imread(im_name)
                    b,g,r = cv2.split(im)
                    im = cv2.merge([r,g,b])
                else:
                    # st.write('Rendering image')
                    # im = render_step_simple(os.path.join(model_folder, filename), offscreen_renderer = renderer)
                    im = render_step_simple(os.path.join(model_folder, filename))

                # Create columns and add image and sim score
                c = st.beta_columns(3)
                c[0].image(im, width = 100)
                c[1].write(os.path.splitext(filename)[0])
                c[2].write(score)

        st.balloons()



if __name__ == "__main__":
    run_it()

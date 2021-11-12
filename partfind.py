# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:41:06 2021

@author: prctha
"""

# HR 04/03/21
# To plot rendered images of parts alongside similarity scores

# streamlit test

import streamlit as st

import pandas as pd
import os
import cv2
import torch
from model.model import GCNTriplet
from CADDataset import CADDataset
from torch_geometric.data import DataLoader
import pickle
import numpy as np
from pathlib import Path
import shutil
import csv
# import OCC
# from OCC.Display.OCCViewer import Viewer3d

from step2image import render_step, render_step_simple
from step2graph import StepToGraph



def load_from_step(step_file):
    print("loading:",step_file)
    
    if isinstance(step_file, str):
        tmp_file = os.path.join('./tmp/',os.path.basename(step_file))
        print("step_file",step_file)
        print("tmp_file",tmp_file)
        tmpPickle = os.path.join("./tmp/raw/",os.path.basename(step_file)+".gz")
        shutil.copyfile(step_file, tmp_file)
        filename = os.path.basename(step_file)
    else:
        bytesData = step_file.getvalue()
        tmp_file = os.path.join('./tmp/',step_file.name)
        f = open(tmp_file,'wb')
        f.write(bytesData)
        f.close()
        tmpPickle = os.path.join("./tmp/raw/",step_file.name+".pickle")
        filename = step_file.name
    
    tmpdataset = {}
    s2g = StepToGraph(tmp_file)
    s2g.compute_faces_surface()
    
    tmpdataset[filename+"a"] = {"cat":0, "graph_nx":s2g.G}
    
    
    if not os.path.exists("./tmp/raw/"):
        os.makedirs("./tmp/raw/")
    if not os.path.exists("./tmp/processed/"):
        os.makedirs("./tmp/processed/")
    print("writing pickle")
    with open(tmpPickle,'wb') as handle:
        pickle.dump(tmpdataset,handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("processing!")
    tmpdataset = CADDataset(".\\tmp\\", filename + ".pickle",
                     force_reprocess=False, add_cats=False)
    print("tmpdataset complete")
    
    
    
    return tmpdataset, filename
    


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu") 


def load_model(convtype="GraphConv"):
    dataset = CADDataset(".\\Dataset\\", "cakebox_nx.pickle",
                     force_reprocess=False,add_cats=False)
    print("ConvType:", convtype)
    model = GCNTriplet(hidden_channels=32,
                   dataset=dataset,
                   nb_layers=3,
                   convtype=convtype).to(device)
    model_path = '.\\saved_models\\partfind_v2_3layers_32_' + convtype + '_40.pt'

    
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    
    
    # if not os.path.exists("demo.pkl"):
    batch_size = 16
    cakebox_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # array = np.empty((0, 16), int)
        # with torch.no_grad():
            # for i, (data, _, _) in enumerate(cakebox_loader):
                # print("i",i)
                # data = data.to(device)

                # if convtype in ['GCNConv', 'GraphConv']:  # node features only
                    # kwargs = {"x0": data.x, "edge_index0": data.edge_index, "batch0": data.batch,
                              # "x1": data.x, "edge_index1": data.edge_index, "batch1": data.batch,
                              # "x2": data.x, "edge_index2": data.edge_index, "batch2": data.batch}
                # elif convtype in ['NNConv']:  # node and edge features
                    # kwargs = {"x0": data.x, "edge_index0": data.edge_index, "batch0": data.batch, "edge_attr0": data.edge_attr,
                              # "x1": data.x, "edge_index1": data.edge_index, "batch1": data.batch, "edge_attr1": data.edge_attr,
                              # "x2": data.x, "edge_index2": data.edge_index, "batch2": data.batch, "edge_attr2": data.edge_attr}

                # Perform a single forward pass for each model.
                # out0, out1, out2, correct, score_p, score_n, correct_s = model(**kwargs)
                # array = np.append(array, np.array(out0), axis=0)
        # print(array.shape)
        
        # rowNames = []
        # with open(".\\dataset\\triple_list.csv", 'r') as f:
            # reader = csv.reader(f, delimiter='\n')
            # for row in reader:
                # if row:
                    # rowNames.append(row[0])
        
        # header_row = ['file']
        # for i in range(0, array.shape[1]):
            # s = "v" + str(i)
            # header_row.append(s)
        
        # df = pd.DataFrame(array, columns=header_row[1:])
        # df['name'] = rowNames
        
        # types = []
        # with open("partTypes.tsv", 'r') as f:
            # reader = csv.reader(f, delimiter='\t')
            # for i, row in enumerate(reader):
                # types.append(row)

        # l = len(rowNames)
        # processes = np.zeros(83)
        # df['process'] = processes


        # for row in types:
            # try:
                # idx = rowNames.index(row[0])
                # df.process[idx] = row[1]
            # except:
                # print("not in list:", row[0])
            
        # with open("demo.pkl", 'wb') as f:
            # pickle.dump(df, f)
    # else:
        # df = pd.read_pickle("./demo.pkl")
    
    return model, cakebox_loader

convtype="GraphConv"
model, cakebox_loader = load_model(convtype=convtype)
print("model loaded")


#To do generate vectors in model
# def generate_cakebox(model):
    



step_folder = ".\cakebox_parts"
image_folder = step_folder
save_folder = ".\save_data"


st.title('Part finder')
st.write("Upload part to find similar:")
uploaded_file = st.file_uploader("Choose a file")
                     
if uploaded_file is not None:
  csv_loc = os.path.join(save_folder,uploaded_file.name) +".csv"

  load_csv=False
  try:
    prev_scores = pd.read_csv(csv_loc).set_index('file')
    scores_dict = prev_scores.to_dict('index')
    # print("scores_dict:",scores_dict)
    load_csv=True
    print("loading from saved results")
  except:
    pass

  st.spinner()

  with st.spinner(text="Rendering file..."):
    image0 = render_step(uploaded_file, remove_tmp = False)
    st.image(image0,caption="Input model",width=400)



  st.write("Finding similar models:")
  with st.spinner(text="Finding results..."):

    tmpdataset, testfile = load_from_step(uploaded_file)

    
    tmploader_loader = DataLoader(tmpdataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for i, (data, _, _) in enumerate(tmploader_loader):
            data = data.to(device)

            if convtype in ['GCNConv', 'GraphConv']:  # node features only
                kwargs = {"x0": data.x, "edge_index0": data.edge_index, "batch0": data.batch,
                          "x1": data.x, "edge_index1": data.edge_index, "batch1": data.batch,
                          "x2": data.x, "edge_index2": data.edge_index, "batch2": data.batch}
            elif convtype in ['NNConv']:  # node and edge features
                kwargs = {"x0": data.x, "edge_index0": data.edge_index, "batch0": data.batch, "edge_attr0": data.edge_attr,
                          "x1": data.x, "edge_index1": data.edge_index, "batch1": data.batch, "edge_attr1": data.edge_attr,
                          "x2": data.x, "edge_index2": data.edge_index, "batch2": data.batch, "edge_attr2": data.edge_attr}

            # Perform a single forward pass for each model.
            out0, out1, out2, correct, score_p, score_n, correct_s = model(**kwargs)
            
        vector = np.array(out0)
        
        array = np.empty((0, 16), int)
        for i, (data, _, _) in enumerate(cakebox_loader):
            print("i",i)
            data = data.to(device)

            if convtype in ['GCNConv', 'GraphConv']:  # node features only
                kwargs = {"x0": data.x, "edge_index0": data.edge_index, "batch0": data.batch,
                          "x1": data.x, "edge_index1": data.edge_index, "batch1": data.batch,
                          "x2": data.x, "edge_index2": data.edge_index, "batch2": data.batch}
            elif convtype in ['NNConv']:  # node and edge features
                kwargs = {"x0": data.x, "edge_index0": data.edge_index, "batch0": data.batch, "edge_attr0": data.edge_attr,
                          "x1": data.x, "edge_index1": data.edge_index, "batch1": data.batch, "edge_attr1": data.edge_attr,
                          "x2": data.x, "edge_index2": data.edge_index, "batch2": data.batch, "edge_attr2": data.edge_attr}

            #Perform a single forward pass for each model.
            out0, out1, out2, correct, score_p, score_n, correct_s = model(**kwargs)
            array = np.append(array, np.array(out0), axis=0)
        
        
    # st.write(vector)
    
    rowNames = []
    with open(".\\dataset\\triple_list.csv", 'r') as f:
        reader = csv.reader(f, delimiter='\n')
        for row in reader:
            if row:
                rowNames.append(row[0])
    
    header_row = ['file']
    for i in range(0, array.shape[1]):
        s = "v" + str(i)
        header_row.append(s)
    
    df = pd.DataFrame(array, columns=header_row[1:])
    df['name'] = rowNames
    
    types = []
    with open("partTypes.tsv", 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(reader):
            types.append(row)

    l = len(rowNames)
    processes = np.zeros(83)
    df['process'] = processes


    for row in types:
        try:
            idx = rowNames.index(row[0])
            df.process[idx] = row[1]
        except:
            print("not in list:", row[0])
    
    #df = pd.read_pickle("./cakeboxVectors/cakebox_vectors_32_" + convtype + "_40_wlin.pkl")
    dud = []
    for index, row in df.iterrows():
        if row['process'] == 0.0:
            print("extra zero!")
            dud.append(index)
        if row['process'] == 'other (not applicable)':
            dud.append(index)
    df = df.drop(dud)
    partlist = []
    distlist = []
    print(testfile[:-5])
    for index, row in df.iterrows():
        
        dist = np.linalg.norm(vector - row[:16].to_numpy())
        if row['name'] == testfile[:-5]:
            print("dist:",dist)
        else:
            partlist.append(row['name']) 
            distlist.append(dist)

    
    
    # # Open renderer to avoid new renderer being opened for each part image
    # renderer = Viewer3d()
    # renderer.Create()
    # renderer.SetSize(600,600)
    # renderer.SetModeShaded()

    file_dict = dict(sorted(zip(distlist, partlist), reverse = False))


    c_head = st.columns(3)
    c_head[0].write('Shape image')
    c_head[1].write('Shape name')
    c_head[2].write('Ranking')
    ranking = 0
    for k,v in file_dict.items():
        ranking += 1
        filename = v
        score = k
        
        # Get/create image
        im_name = os.path.splitext(filename)[0] + ".jpg"
        im_name = os.path.join(image_folder, im_name)
        if os.path.isfile(im_name):
            # st.write('Found image')
            im = cv2.imread(im_name)
            b,g,r = cv2.split(im)
            im = cv2.merge([r,g,b])
        else:
            # st.write('Rendering image')
            # im = render_step_simple(os.path.join(model_folder, filename), offscreen_renderer = renderer)
            im = render_step_simple(os.path.join(image_folder, filename))

        # Create columns and add image and sim score
        c = st.columns(3)
        c[0].image(im, width = 100)
        c[1].write(os.path.splitext(filename)[0])
        c[2].write(ranking)



    # delete temp files
    dirpath = Path('./tmp/processed/')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    dirpath = Path('./tmp/raw/')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

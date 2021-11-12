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
import cv2

#from step2image import makeSnapshotWithoutGui

image_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Image"
tmp_folder = 'C:\\Users\\prctha\\PythonDev\\partfind\\tmp\\'
st.title('Part finder (mockup)')

import sys
#st.write(sys.version)
st.write("Upload part to find similar:")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  # find image
  image_file0 = os.path.splitext(uploaded_file.name)[0]
  image_file0 = os.path.join(image_folder,image_file0) + ".png"
  image0 = cv2.imread(image_file0)
  
  st.image(image0,caption="Input model",width=400)
  
  #st.write("Rendering file")
#  makeSnapshotWithoutGui(uploaded_file)
  #st.write("File rendered")
#  image_file0a = os.path.splitext(uploaded_file.name)[0]
#  image0a = cv2.imread(os.path.join(tmp_folder,image_file0a) + ".png")
  
  #st.image(image0a,caption="Rendered model",width=400)
  
  
  st.write("Finding similar models:")
  
  # # at the moment pick 3 files in folder at random
  image_file1 = random.choice(os.listdir(image_folder)) #change dir name to whatever
  image_file2 = random.choice(os.listdir(image_folder)) #change dir name to whatever
  image_file3 = random.choice(os.listdir(image_folder)) #change dir name to whatever
  image_file1 = os.path.join(image_folder,image_file1)
  image_file2 = os.path.join(image_folder,image_file2)
  image_file3 = os.path.join(image_folder,image_file3)
  image1 = cv2.imread(image_file1)
  image2 = cv2.imread(image_file2)
  image3 = cv2.imread(image_file3)
  images = [image1,image2,image3]
  st.image(images,caption=["Found model 1","Found model 2","Found model 3"],width= 300)
  
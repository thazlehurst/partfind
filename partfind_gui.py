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
try:
	import OCC
	st.write("OCC loaded")
except:
	st.write("OCC not loaded")
from step2image_pyocc import render_step

#from step2image import makeSnapshotWithoutGui

#image_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Image"
model_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data"
tmp_folder = 'C:\\Users\\prctha\\PythonDev\\partfind\\tmp\\'
st.title('Part finder (mockup)')

import sys


st.write("Upload part to find similar:")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  # find image
  image_file0 = os.path.splitext(uploaded_file.name)[0]
  image_file0 = os.path.join(image_folder,image_file0) + ".png"

  st.spinner()
  with st.spinner(text="Rendering file..."):
    image0 = render_step(uploaded_file)
    st.image(image0,caption="Input model",width=400)
  
  
  
  st.write("Finding similar models:")
  
  # # at the moment pick 3 files in folder at random
  with st.spinner(text="Finding results..."):
    size = 100000000
    while size > 500000:
      size = 0
      model_file1 = random.choice(os.listdir(model_folder)) #change dir name to whatever
      model_file2 = random.choice(os.listdir(model_folder)) #change dir name to whatever
      model_file3 = random.choice(os.listdir(model_folder)) #change dir name to whatever
      model_file1 = os.path.join(model_folder,model_file1)
      model_file2 = os.path.join(model_folder,model_file2)
      model_file3 = os.path.join(model_folder,model_file3)
      size = os.stat(model_file1).st_size
      #st.write(os.stat(model_file1).st_size)
      #st.write(os.stat(model_file2).st_size)
      size = max(os.stat(model_file2).st_size,size)
      #st.write(os.stat(model_file3).st_size)
      size = max(os.stat(model_file3).st_size,size)


    image1 = render_step(model_file1)
    image2 = render_step(model_file2)
    image3 = render_step(model_file3)
    images = [image1,image2,image3]
    
    for i in range(0,3): # for 3 rows
      cols = st.beta_columns(2) # 2 columns
      cols[0].image(images[i],width=300)
      result_str = "Match:" + str(random.randint(1,101)) + "%"
      cols[1].write(result_str)
  #st.image(images,caption=["Found model 1","Found model 2","Found model 3"],width= 300)
  
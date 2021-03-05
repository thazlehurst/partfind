## Script for taking input step file, and exporting a isometric rendering of it as an image using pythonocc


import sys
import os
import shutil
import cv2
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Extend.DataExchange import read_step_file
#from OCC.Extend.DataExchange import read_iges_file
import numpy

tmp = 'C:\\Users\\prctha\\PythonDev\\partfind\\tmp\\'


def step2image(filename):
	print("filename:",filename)
	# igesfiles = "/nobackup/prctha/CadGrabData/Models/IgsFiles/"
	# igesimages = "/nobackup/prctha/CadGrabData/Models/IgsImages/"
	# stepfilename = os.path.basename(filename)
	# pngfilename = os.path.join(tmp,stepfilename) + ".png"


def render_step(step_file,remove_tmp=True):
	# need to create temp file
	# print("type",type(step_file.read()))
	# tmp_file = os.path.join('./tmp/',step_file.name)
	# f = open(tmp_file,'wb')
	# print(step_file.read())
	# f.write(step_file.read())
	# f.close()
	if isinstance(step_file, str):
		tmp_file = os.path.join('./tmp/',os.path.basename(step_file))
		print("step_file",step_file)
		print("tmp_file",tmp_file)
		shutil.copyfile(step_file, tmp_file)
	else:
		bytesData = step_file.getvalue()
		tmp_file = os.path.join('./tmp/',step_file.name)
		f = open(tmp_file,'wb')
		f.write(bytesData)
		f.close()

	# create the renderer
	size = 600
	offscreen_renderer = Viewer3d() # None
	offscreen_renderer.Create()
	offscreen_renderer.SetSize(size, size)
	offscreen_renderer.SetModeShaded()


	shp = read_step_file(tmp_file)

	offscreen_renderer.DisplayShape(shp, update=True) #color="orange"

	# send the shape to the renderer
	#change view point
	cam = offscreen_renderer.View.Camera()

	center = cam.Center()
	eye = cam.Eye()

	#start_display()
	data = offscreen_renderer.GetImageData(size, size) # 1

	#set background colour
	offscreen_renderer.set_bg_gradient_color([255,255,255],[255,255,255])

	# set ray tracing
	offscreen_renderer.SetRaytracingMode(depth=3)

	#eye.SetY(eye.Y() + 45)
	#cam.SetEye(eye)
	offscreen_renderer.View.ZFitAll()
	offscreen_renderer.Context.UpdateCurrentViewer()

	data = offscreen_renderer.GetImageData(size, size) # 1
	png_name = os.path.splitext(tmp_file)[0] + ".png" # step_file.name
	# export the view to image
	tmp_png = png_name #os.path.join('./tmp/',png_name)
	offscreen_renderer.View.Dump(tmp_png)
#    offscreen_renderer.View.Dump(testout)

	image = cv2.imread(tmp_png,3)
	b,g,r = cv2.split(image)           # get b, g, r
	image = cv2.merge([r,g,b])

	# delete tmp files
	if remove_tmp:
		os.remove(tmp_png)
	os.remove(tmp_file)

	return image



def render_step_simple(step_file, offscreen_renderer = None, size = 600):

    if not offscreen_renderer:
        offscreen_renderer = Viewer3d()
        offscreen_renderer.Create()
        offscreen_renderer.SetModeShaded()
        offscreen_renderer.SetSize(size, size)

    shp = read_step_file(step_file)
    offscreen_renderer.EraseAll()
    
    offscreen_renderer.DisplayShape(shp, update=True) #color="orange"

    offscreen_renderer.set_bg_gradient_color([255,255,255],[255,255,255])
    offscreen_renderer.SetRaytracingMode(depth=3)

    offscreen_renderer.View.FitAll()
    offscreen_renderer.View.ZFitAll()
    # offscreen_renderer.Context.UpdateCurrentViewer()

    im_name = os.path.splitext(step_file)[0] + ".png"
    offscreen_renderer.View.Dump(im_name)

    image = cv2.imread(im_name,3)
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])

    return image


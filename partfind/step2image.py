import sys
#sys.path.append('C:\\Users\\prctha\\AppData\\Local\\Continuum\\anaconda3\\envs\\fc\\Library\\bin')
#sys.path.append('C:\\Users\\prctha\\Anaconda3\\envs\\partfind\\Library\\bin')

sys.path.append('C:\\Users\\prctha\\AppData\\Local\\FreeCAD 0.18\\bin')

import FreeCAD
import Part
import numpy as np
import Mesh
from pivy import coin
import os

COIN_FULL_INDIRECT_RENDERING=1

def light(x,y,z,c1,c2,c3,i):
  light = coin.SoDirectionalLight()
  light.on = True
  light.color = (c1,c2,c3)
  light.intensity = i
  light.direction = (x,y,z)
  return light

def makeSnapshotWithoutGui(stepfilename): # full path
    stepfilename = os.path.basename(stepfilename)
    # determine output file names
    ivfilename = os.path.splitext(stepfilename)[0] + ".iv"
    ivfilename = os.path.join('C:\\Users\\prctha\\PythonDev\\partfind\\tmp\\',ivfilename)
    print(ivfilename)
#    psfilename = os.path.splitext(stepfilename)[0] + ".ps"
    pngfilename = os.path.splitext(stepfilename)[0] + ".png"
    print(pngfilename)
    pngfilename = os.path.join('C:\\Users\\prctha\\PythonDev\\partfind\\tmp\\',pngfilename)
    print(pngfilename)
#    pngfilename = "test2.png"

    # open a STEP file
    print(stepfilename)
    shape=Part.read(stepfilename)
    #shape=mesh
    print(ivfilename)
    f=open(ivfilename,"w")
    f.write(shape.writeInventor())
    f.close()

    # convert to iv format
    inp=coin.SoInput()
    inp.openFile(ivfilename)

    # and create a scenegraph
    data = coin.SoDB.readAll(inp)
    base = coin.SoBaseColor()
    base.rgb.setValue(0.9,0.7,1.0)
    data.insertChild(base,0)

    # add light and camera so that the rendered geometry is visible
    root = coin.SoSeparator()
    # light = coin.SoDirectionalLight()
    lig = light(-1,0,-1,1,1,1,1)
    lig2 = light(1,1,-2,0.7,0.7,0.7,0.7)
    cam = coin.SoOrthographicCamera()
    root.addChild(cam)
    root.addChild(lig)
    root.addChild(lig2)
    root.addChild(data)

    # do the rendering now
    axo = coin.SbRotation(-0.353553, -0.146447, -0.353553, -0.853553)
    viewport=coin.SbViewportRegion(512,512)
    cam.orientation.setValue(axo)
    cam.viewAll(root,viewport)
    off=coin.SoOffscreenRenderer(viewport)

    # background color
    bgColor = coin.SbColor(0.8, 0.8, 0.8)
    off.setBackgroundColor(bgColor)

    root.ref()
    off.render(root)
    root.unref()

    # export the image, PS is always available
#    off.writeToPostScript(psfilename)

    # Other formats are only available if simage package is installed
    if off.isWriteSupported("PNG"):
        off.writeToFile(pngfilename,"PNG")
#        
    # delete temp file
    os.remove(ivfilename)

#sys.argv[1]
makeSnapshotWithoutGui(sys.argv[1])
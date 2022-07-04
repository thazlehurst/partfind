# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:01:50 2022

@author: prctha
"""
import OCC
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.Bnd import Bnd_Box

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties

def extract_info(file_name):
    # this script loads a step file and returns the bounding box, the surface area and the volume
    # Read the file and get the shape
    reader = STEPControl_Reader()
    tr = reader.WS().TransferReader()
    reader.ReadFile(file_name)
    reader.TransferRoots()
    shp = reader.OneShape()
    
    bb = get_boundingbox(shp)
    print("bounding box:", bb[6:9])
    
    surf = shape_faces_surface(shp)
    print("Surface area",surf)
    
    vol = get_volume(shp)
    print("volume",vol)
    
    

def get_boundingbox(shape, tol=1e-6, use_mesh=True):
    """return the bounding box of the TopoDS_Shape `shape`
    Parameters
    ----------
    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from
    tol: float
        tolerance of the computed boundingbox
    use_mesh : bool
        a flag that tells whether or not the shape has first to be meshed before the bbox
        computation. This produces more accurate results
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    if use_mesh:
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        if not mesh.IsDone():
            raise AssertionError("Mesh not done.")
    brepbndlib_Add(shape, bbox, use_mesh)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return xmin, ymin, zmin, xmax, ymax, zmax, xmax - xmin, ymax - ymin, zmax - zmin
    
def shape_faces_surface(shape):
    """Compute the surface of each face of a shape"""
    # then loop over faces
    t = TopologyExplorer(shape)
    props = GProp_GProps()
    shp_idx = 1
    total_surface = 0
    for face in t.faces():
        brepgprop_SurfaceProperties(face, props)
        face_surf = props.Mass()
        total_surface = total_surface + face_surf
        #print("Surface for face nbr %i : %f" % (shp_idx, face_surf))
        #shp_idx += 1
    
    return total_surface
    
def get_volume(shape):
    tolerance = 1e-5 # Adjust to your liking
    props = GProp_GProps()
    volume = brepgprop_VolumeProperties(shape, props, tolerance)
    
    return volume

if __name__ == "__main__":
    test_file = 'C:\\Users\\prctha\\PythonDev\\ABC_Data\\00000006_d4fe04f0f5f84b52bd4f10e4_step_001.step'
    #test_file = 'C:\\Users\\prctha\\PythonDev\\ABC_Data\\00000064_767e4372b5f94a88a7a17d90_step_005.step'
    #test_file = 'C:\\Users\\prctha\\PythonDev\\ABC_Data\\00000374_23c957e8e7b2428282a13ae2_step_007.step'
    #test_file = 'C:\\Users\\prctha\\Desktop\\demo models\\0000024773.STEP'
    print("test file:",test_file)
    extract_info(test_file)

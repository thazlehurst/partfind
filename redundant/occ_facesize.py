from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Display.OCCViewer import Viewer3d
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties


# for working out what type of surface everything is
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface, GeomAbs_OtherSurface 

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID, TopAbs_ShapeEnum)
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.TCollection import TCollection_HAsciiString

try:
	import OCC
	#st.write("OCC loaded")
except:
	st.write("OCC not loaded")
    
def shape_faces_surface():
    """ Compute the surface of each face of a shape
    """
    # first create the shape
    the_shape = BRepPrimAPI_MakeBox(50., 30., 10.).Shape()
    # then loop over faces
    t = TopologyExplorer(the_shape)
    props = GProp_GProps()
    shp_idx = 1
    for face in t.faces():
        brepgprop_SurfaceProperties(face, props)
        face_surf = props.Mass()
        print("Surface for face nbr %i : %f" % (shp_idx, face_surf))
        shp_idx += 1


def compute_faces_surface(filename):
    """ Compute surface area of faces from step file"""
    
    # create the renderer
    size = 600
    offscreen_renderer = Viewer3d() # None
    offscreen_renderer.Create()
    offscreen_renderer.SetSize(size, size)
    
    #shp = read_step_file(filename)
    # Read the file and get the shape
    reader = STEPControl_Reader()
    tr = reader.WS().TransferReader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shp = reader.OneShape()
    
    # exp = TopExp_Explorer(shp,TopAbs_FACE)
    t = TopologyExplorer(shp)
    
    props = GProp_GProps()
    shp_idx = 1

    for face in t.faces():
        brepgprop_SurfaceProperties(face, props)
        face_surf = props.Mass()
        print("Surface for face nbr %i : %f" % (shp_idx, face_surf))
        
        surf = BRepAdaptor_Surface(face, True)
        surf_type = surf.GetType() # number corresponds to GeomAbs_surfaceType
        # if surf_type == GeomAbs_Plane:
            # print("this surface is a plane")
        print("surf_type",surf_type)
        item = tr.EntityFromShapeResult(face, 1)
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.SetName(TCollection_HAsciiString(str(shp_idx)))
        #print("name",name)
        shp_idx += 1
    
    
    edge_idx = 1
    for edge in t.edges():
        print("edge",edge_idx)
        for face in t.faces_from_edge(edge):
            item = tr.EntityFromShapeResult(face, 1)
            item = StepRepr_RepresentationItem.DownCast(item)
            name = item.Name().ToCString()
            print("name",name)
        edge_idx += 1
       #print("faces from edge",t.faces_from_edge(edge))
        
        
    object_methods = [method_name for method_name in dir(item)
                  if callable(getattr(item, method_name))]
    print("object_methods",object_methods)
    #print("props",props)

test_file = 'C:\\Users\\prctha\\PythonDev\\ABC_Data\\00000006_d4fe04f0f5f84b52bd4f10e4_step_001.step'

#shape_faces_surface()
compute_faces_surface(test_file)

# to do, make it a networkx/dgl graph
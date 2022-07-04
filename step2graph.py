from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Display.OCCViewer import Viewer3d
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.BRepGProp import brepgprop_LinearProperties,BRepGProp_EdgeTool,BRepGProp_EdgeTool_Value
from OCC.Core.TopoDS import topods_Edge, topods_Face

# for working out what type of surface everything is
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface, GeomAbs_OtherSurface 
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.Geom import Geom_Line
from OCC.Extend.ShapeFactory import make_vertex, make_edge

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf, gp_Lin, gp_Dir

from OCC.Core.Quantity import (Quantity_Color, Quantity_TOC_RGB)

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID, TopAbs_ShapeEnum, TopAbs_REVERSED, TopAbs_FORWARD, 
                             TopAbs_INTERNAL , TopAbs_EXTERNAL, TopAbs_Orientation) 
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.TCollection import TCollection_HAsciiString

from OCCUtils.face import Face

import networkx as nx
import matplotlib.pyplot as plt
from itertools import count
import sys
import math

class StepToGraph:
    def __init__(self,file_name):
        self.file_name = file_name
        self.create_graph = True
        try:
            import OCC
            #print("OCC loaded")
        except:
            print("OCC not loaded")
            
        self.face_props = {}
        self.edge_props = {}
    
    def set_graph(self):
        if self.create_graph:
            print("Not saving graph")
        else:
            print("Saving graph")
        self.create_graph = not self.create_graph
    
    
    def compute_faces_surface(self):
        """ Compute surface area of faces from step file"""
        
        # create the renderer
        size = 600
        offscreen_renderer = Viewer3d() # None
        offscreen_renderer.Create()
        offscreen_renderer.SetSize(size, size)
        
        # Read the file and get the shape
        reader = STEPControl_Reader()
        tr = reader.WS().TransferReader()
        reader.ReadFile(self.file_name)
        reader.TransferRoots()
        self.shp = reader.OneShape()

        t = TopologyExplorer(self.shp)
        
        props = GProp_GProps()
        shp_idx = 0
        
        if self.create_graph:
            self.G = nx.DiGraph()
        
        for face in t.faces():
            self.face_props[face] = shp_idx
            brepgprop_SurfaceProperties(face, props)
            faceSurfArea = props.Mass() # surface_area

            surf = BRepAdaptor_Surface(face, True)
            faceSurfType = surf.GetType() # number corresponds to GeomAbs_surfaceType
            
            item = tr.EntityFromShapeResult(face, 1)
            item = StepRepr_RepresentationItem.DownCast(item)
            name = item.SetName(TCollection_HAsciiString(str(shp_idx)))
            #print("name",name)
            #print(face.Convex())
            if self.create_graph:
                self.G.add_node(shp_idx,surfArea=faceSurfArea,surfType=faceSurfType)
            
            shp_idx += 1
        
        
        normalise = False
        
        if normalise:
            d = nx.get_node_attributes(self.G,"surfArea")
            surfAreas = [v * 100 for v in d.values()]
            norm = [float(i)/sum(surfAreas) for i in surfAreas]
            for node, i  in enumerate(d):
                self.G.nodes[node]['surfArea'] = norm[i]
        
        
        # instead of looking at each edge, look at each edge per face
        
        
        
        
        self.vects = {}
        
        for face in t.faces():
            #print("face:", self.face_props[face])
            # for each face, get its edges
            self.face_props[face] = face.Orientation()
            edges = TopExp_Explorer(face,TopAbs_EDGE)
            #for each edge on this face
            while edges.More():
                edge = topods_Edge(edges.Current())
                
                curve = BRepAdaptor_Curve(edge)
                length = props.Mass() ## needs changing, calculate whole face perimeter, then have length as faction of total
                
                # for each each find the two faces (or face) that connect them
                face_n = 0
                num_faces_connected = len(list(t.faces_from_edge(edge)))

                assert num_faces_connected < 3, "Edge connected to more than two faces"
                assert num_faces_connected > 0, "Edge connected to no faces"
                
                #print("num fces connected",num_faces_connected)
                # gets faces for that edge
                for edge_face in t.faces_from_edge(edge):
                    face_n += 1
                    item = tr.EntityFromShapeResult(edge_face, 1)
                    item = StepRepr_RepresentationItem.DownCast(item)
                    name = item.Name().ToCString()
                    
                    if edge_face == face:
                        face_1 = int(name)
                        S1 = BRep_Tool().Surface(edge_face)
                    else:
                        face_2 = int(name)
                        S2 = BRep_Tool().Surface(edge_face)
                        face_adj = edge_face
                if num_faces_connected == 1:
                    face_2 = face_1
                    S2 = S1
                    face_adj = face
                
                # Get vector along edge for face F (current face)
                
                try: # sometimes num_faces_connected = 1, and this causes BRep_Tool to not work correctly
                    CurveHandle, f, l = BRep_Tool().Curve(edge)
                
                    midParam = (f + l)*0.5
                    paramStep = (l - f)*0.1
                    A_param = midParam - paramStep
                    B_param = midParam + paramStep
                    
                    
                    A = CurveHandle.Value(A_param)
                    B = CurveHandle.Value(B_param)
                
                    # Calculate vector F
                    # V_x
                    Vx = gp_Vec(B,A)
                    
                    assert Vx.Magnitude() > sys.float_info.epsilon, "Error, angle undefined"
                    # if pointing wrong way, reverse
                    if edge.Orientation() == TopAbs_FORWARD:#TopAbs_REVERSED:
                        Vx.Reverse()
                    
                    S1 = BRep_Tool().Surface(face)
                    SAS = ShapeAnalysis_Surface(S1)
                    UV = SAS.ValueOfUV(A,1.0e-1)
                    
                    P = gp_Pnt()
                    D1U = gp_Vec()
                    D1V = gp_Vec()
                    S1.D1(UV.X(),UV.Y(),P,D1U,D1V)
                    
                    N = D1U.Crossed(D1V)  #normal vector
                    
                    # make sure normal is pointing out
                    if face.Orientation() == TopAbs_REVERSED:
                        N.Reverse()
                    
                    FP = P
                    FN = N
                    
                    #V_y
                    Vy = N.Crossed(Vx)
                    assert Vy.Magnitude() > sys.float_info.epsilon, "Error, angle undefined"
                    
                    TF = Vy.Normalized() ### tangent for F
                    Ref = Vx.Normalized() ##F Vx
                    
                    ## do same for G
                    # Calculate vector for G
                    # V_x
                    Vx = gp_Vec(B,A)
                    assert Vx.Magnitude() > sys.float_info.epsilon, "Error, angle undefined"
                    if edge.Orientation() ==  TopAbs_REVERSED:# TopAbs_FORWARD: # If forward, while makes more sense, a flat join is 180 degrees, where I want that to be zero 
                        Vx.Reverse()
                    
                    #CumulOriOnF 
                    #V_z (normal)
                    SAS = ShapeAnalysis_Surface(S2)
                    UV = SAS.ValueOfUV(A,1.0e-1)
                
                    P = gp_Pnt()
                    D1S = gp_Vec()
                    D1T = gp_Vec()
                    S2.D1(UV.X(),UV.Y(),P,D1S,D1T)
                
                    N = D1S.Crossed(D1T)
                    # make sure normal is pointing out
                    if face_adj.Orientation() == TopAbs_REVERSED:
                        N.Reverse()
                    
                    GP = P
                    GN = N
                
                    #V_y
                    Vy = N.Crossed(Vx)
                    assert Vy.Magnitude() > sys.float_info.epsilon, "Error, angle undefined"
                
                    TG = Vy.Normalized() ### tangent for G
                    
                    angleRad = TF.AngleWithRef(TG,Ref)
                    if angleRad < 0:
                        angleRad = -math.pi - angleRad
                    else:
                        angleRad = math.pi - angleRad
                    degrees = math.degrees(angleRad)
                    
                    
                    
                    
                    self.vects[edge] = {"Ref":Ref,"TF":TF,"FP":FP,"FN":FN,"GP":P,"TG":TG,"GN":GN,"degrees":degrees,"angleRad":angleRad}
                except:
                    print("num_faces_connected",num_faces_connected)
                    print("skipped edge")
                    # print("edge??",edge.IsNull())
                    # print("or",edge.Orientation())
                    # CurveHandle, f = BRep_Tool().Curve(edge)
                    # print("CH",CurveHandle)
                    # print("f",f)
                curveType = curve.GetType()
                def getRelSize(face_a,face_b):
                    surfA = self.G.nodes[face_a]['surfArea']
                    surfB = self.G.nodes[face_b]['surfArea']
                    return surfA/surfB, surfB/surfA
                
                if self.create_graph:
                    rel1, rel2 = getRelSize(face_1,face_2)
                    length = length
                    self.G.add_edge(face_1,face_2,length=length,relativeSize=rel1,edgeType=curveType,degrees=degrees)
                
                
                
                edges.Next()
            
        
            
             ## http://quaoar.su/blog/page/issledovanie-telesnogo-ugla
             ### https://gitlab.com/ssv/AnalysisSitus/-/blob/master/src/asiAlgo/features/asiAlgo_CheckDihedralAngle.cpp
             ## line 217
            
        
        
        if normalise:
            l = nx.get_edge_attributes(self.G,"length")
            #print("l",l)
            ls = [v for v in l.values()]
            suml = sum(ls)
            for u,v,a in self.G.edges(data=True):
                #print(u,v,a)
                self.G.edges[u,v]['length'] = float(a['length'])/suml
            
        
        # object_methods = [method_name for method_name in dir(S1)
        #               if callable(getattr(S1, method_name))]
        # print("object_methods SAS",object_methods)
        #print("props",props)
        
        #if self.create_graph:
        #    self.plot_graph()
            
    def plot_graph(self):
        print("Plotting...")
        #print("self.G.nodes",self.G.nodes)
        
        # for surface area of nodes
        d = nx.get_node_attributes(self.G,"surfArea")
        
        # colour of nodes for type of node
        # get unique groups
        groups = set(nx.get_node_attributes(self.G,'surfType').values())
        print("Groups",groups)
        mapping = dict(zip(sorted(groups),count()))
        nodes = self.G.nodes()
        colors = [mapping[self.G.nodes[n]['surfType']] for n in nodes]

        # change alpha depending on length of edge
        l = nx.get_edge_attributes(self.G,"relativeSize")
        ecolor =[v for v in l.values()]
        ecolor = [float(i)/max(ecolor) for i in ecolor]
        
        # edge type
        groupse = set(nx.get_edge_attributes(self.G,'edgeType').values())
        mappinge = dict(zip(sorted(groupse),count()))
        edges = self.G.edges()
        colorse = [mappinge[self.G.edges[e]['edgeType']] for e in edges]
        
        #edge angle
        # a = nx.get_edge_attributes(self.G,"degrees")
        # ecolor =[v +180 for v in a.values()]
        # ecolor = [float(i)/360 for i in ecolor]
        
        pos = nx.spring_layout(self.G,k=0.22,iterations=500)
        
        #print(max(d.values()))
        scale = 1000*7/max(d.values())
        plt.figure(figsize=(scale,scale))
        nc = nx.draw_networkx_nodes(self.G,pos,
                                    nodelist=d.keys(),
                                    node_size=100,#[v * scale for v in d.values()],
                                    node_color=colors,
                                    cmap=plt.cm.rainbow,)
                                    #with_labels=True)
        
        ec = nx.draw_networkx_edges(self.G,pos,
                                    edgelist=l.keys(),
                                    #edge_color=colorse,
                                    #edge_cmap=plt.cm.rainbow)
                                    edge_color=ecolor,
                                    #edge_cmap=plt.cm.rainbow)
                                    
                                    edge_cmap=plt.cm.binary)
        lc = nx.draw_networkx_labels(self.G,pos)
        plt.show()
        
    def display_shape(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Display.SimpleGui import init_display
        
        display, start_display, add_menu, add_function_to_menu = init_display()
        
        #shp = read_step_file(self.file_name)
        #my_box = BRepPrimAPI_MakeBox(10.,20.,30.).Shape()
        
        
        def line_clicked(shp, *kwargs):
            """ This function is called whenever a line is selected
            """
            t = TopologyExplorer(self.shp)
            def plot_arrow(v,p,ori):
                
                def rgb_color(r, g, b):
                    return Quantity_Color(r, g, b, Quantity_TOC_RGB)
                
                display.DisplayVector(v,p,update=True)
                line = make_vertex(p)#gp_Lin(p,gp_Dir(v))
                q = gp_Pnt(p.X()+v.X(),p.Y()+v.Y(),p.Z()+v.Z())
                line = make_edge(p,q)
                if ori == 0:
                    color=rgb_color(1,0,0)
                elif ori == 1:
                    color=rgb_color(0,1,0)
                else:
                    color=rgb_color(0,0,1)
                
                display.DisplayShape(line,update=True,color=color)
            
            for shape in shp: # this should be a TopoDS_Edge
                print("Edge selected: ", shape)
                e = topods_Edge(shape)
                print("e:",e)
                for face in t.faces_from_edge(e):
                    exp = TopExp_Explorer(face,TopAbs_EDGE)
                    print(exp.Current().Orientation())
                
                #plot_arrow(self.vects[e]['v1'],self.vects[e]['p'])
                #plot_arrow(self.vects[e]['v2'],self.vects[e]['p'])
                #plot_arrow(self.vects[e]['v3'],self.vects[e]['p'])
                
                plot_arrow(self.vects[e]['TF'],self.vects[e]['GP'],0)
                plot_arrow(self.vects[e]['TG'],self.vects[e]['GP'],1)
                plot_arrow(self.vects[e]['Ref'],self.vects[e]['GP'],2)
                
                #print("edge:",self.vects[e]['num'])
                print("rads:",self.vects[e]['angleRad'])
                print("degrees:",self.vects[e]['degrees'])
        
        
        def face_clicked(shp, *kwargs):
            """ This function is called whenever a line is selected
            """
            t = TopologyExplorer(self.shp)
            def plot_arrow(v,p,ori):
                
                def rgb_color(r, g, b):
                    return Quantity_Color(r, g, b, Quantity_TOC_RGB)
                
                display.DisplayVector(v,p,update=True)
                line = make_vertex(p)#gp_Lin(p,gp_Dir(v))
                q = gp_Pnt(p.X()+v.X(),p.Y()+v.Y(),p.Z()+v.Z())
                line = make_edge(p,q)
                if ori == 0:
                    color=rgb_color(1,0,0)
                elif ori == 1:
                    color=rgb_color(0,1,0)
                else:
                    color=rgb_color(0,0,1)
                
                display.DisplayShape(line,update=True,color=color)
            
            for shape in shp: # this should be a TopoDS_Edge
                #print("Face selected: ", shape)
                face = topods_Face(shape)
                #print("Face:",face)
                for edge in t.edges_from_face(face):
                    #print("edge",edge)
                    #exp = TopExp_Explorer(edge,TopAbs_FACE)
                    
                    plot_arrow(self.vects[edge]['TF'],self.vects[edge]['GP'],0)
                    #print(edge.Orientation())
                    plot_arrow(self.vects[edge]['TG'],self.vects[edge]['GP'],1)
                    plot_arrow(self.vects[edge]['Ref'],self.vects[edge]['GP'],2)
                
                    #print("edge:",self.vects[e]['num'])
                    #print("rads:",self.vects[e]['rads'])
                    #print("degrees:",self.vects[e]['degrees'])
        
        display.DisplayShape(self.shp, update=True)
        
        
        
            
        
        # for vect in self.vects:
        #     plot_arrow(vect['v1'],vect['p'])
        #     plot_arrow(vect['v2'],vect['p'])
        #     plot_arrow(vect['v3'],vect['p'])
#        display.DisplayVector(self.v1,self.p,update=True)
#        display.DisplayVector(self.v2,self.p,update=True)
#        display.DisplayVector(self.v3,self.p,update=True)
        display.SetSelectionModeEdge() # switch to edge selection mode
        display.register_select_callback(line_clicked)
        
        #display.SetSelectionModeFace()
        #display.register_select_callback(face_clicked)

        start_display()
    
    def export_graph(self,filename):
        # .gz
        nx.write_gpickle(self.G,filename)
        print("graph saved")


if __name__ == "__main__":
    #test_file = 'C:\\Users\\prctha\\PythonDev\\ABC_Data\\00000006_d4fe04f0f5f84b52bd4f10e4_step_001.step'
    #test_file = 'C:\\Users\\prctha\\PythonDev\\ABC_Data\\00000064_767e4372b5f94a88a7a17d90_step_005.step'
    #test_file = 'C:\\Users\\prctha\\PythonDev\\ABC_Data\\00000374_23c957e8e7b2428282a13ae2_step_007.step'
    test_file = "C:\\Users\\prctha\\PythonDev\\Datasets\\cakebox_parts\\0000024773.step"
    
    A = StepToGraph(test_file)
    #A.set_graph()
    #shape_faces_surface()
    A.compute_faces_surface()
    A.plot_graph()
    #A.display_shape()
    #A.export_graph(filename)
    # to do, make it a networkx/dgl graph
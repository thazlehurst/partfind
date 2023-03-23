# TH March 2020
# Step to Attributed graphs from b-rep modesl

# maybe merge with step_parse at some point

# Regular expression module
import re
import os
# from treelib import Node, Tree
import networkx as nx
from itertools import count
import matplotlib.pyplot as plt
import numpy as np

from EulerOps import EulerOps


class FileReader():  #ner version holds file in memory, loads quicker!
	def __init__(self,step_filename):
		self.filename = step_filename
		self.item_dict = dict({'closed_shell': 'CLOSED_SHELL','open_shell':'OPEN_SHELL'})
		fileHandler = open(self.filename,"r")
		self.listOfLines = fileHandler.readlines()
		fileHandler.close()
		
		
	def find_types(self,type):
		line_hold = ''
		line_type = ''
		type_lines = []
		for line in self.listOfLines:
			index = re.search("#(.*)=", line)
			if index: # if line carries over from previous line in file this wont be a number
				if line_hold:
					if line_type == type:
						type_lines.append(line_hold)
					line_hold = ''
					line_type = ''
				
				prev_index = True
				if self.item_dict[type] in line:
					line_hold = line.rstrip()
					line_type = type
			else:
				prev_index = False
				if 'ENDSEC;' in line:
					if line_hold:
						if line_type == type:
							type_lines.append(line_hold)
						line_hold = ''
						line_type = ''
				else:
					line_hold = line_hold + line.rstrip()
		return type_lines
		
	def find_node(self,node):
		line_hold = ''
		prev_index = False
		node_int = int(node[1:])
		node_lines = []
		#with open(self.filename) as f:
		for line in self.listOfLines:
			index = re.search("#(.*)=", line)
			if index: # if line carries over from previous line in file this wont be a number
				try:
					index_int = int(re.search('[0-9]+', index.string).group())
				except:
					index_int = 0
					pass
				if line_hold:
					node_lines.append(line_hold)
					line_hold = ''
				if ';' not in line:
					prev_index = True
				if node_int == index_int:
					line_hold = line.rstrip()
			else:
				if prev_index == True: ##
					
					if ';' in line:
						if line_hold:
							node_lines.append(line_hold)
							line_hold = ''
							prev_index = False ##
					
					else:
						line_hold = line_hold + line.rstrip()
						return node_lines

class Edge_Bank():
	def __init__(self):
		self.edge_lib = {}
		
	def add_edge(self,edge_ref,face_ref):
		#look in list of egdes if it exists
		if edge_ref in self.edge_lib:
			# edge appears before find corresponding face
			prev_face_ref = self.edge_lib[edge_ref]
			self.edge_lib.pop(edge_ref,None)
			return True, prev_face_ref # returns previous face_ref
		else: #save for later
			self.edge_lib[edge_ref] = face_ref 
			return False, ''

		
class Step_brep():
	def __init__(self):
		self.plot_size = (8,8)
		self.euler = {'v' :0, 'e': 0, 'f': 0, 's': 0, 'h': 0, 'r': 0} # parameters for euler-poincare formula
		
	def euler_valid(self):
		# check if graph is valid solid
		# v - vertices, e - edges, f - faces
		# s - number of disconnected components
		# h - number of holes through the solid
		# r - total number of rings in solid (also known as rings)
		# v - e + f = 2 ( s - h ) + r 
		print('v: ', self.euler['v'])
		print('e: ', self.euler['e'])
		print('f: ', self.euler['f'])
		print('s: ', self.euler['s'])
		print('h: ', self.euler['h'])
		print('r: ', self.euler['r'])
		print("Number of selfloops: ", nx.number_of_selfloops(self.EG))
		print("List of all nodes with self-loops: ", 
			list(nx.selfloop_edges(self.EG,keys=True)))
			#  list(nx.nodes_with_selfloops(self.g))) 
		lhs = self.euler['v'] - self.euler['e'] + self.euler['f']
		rhs = 2*(self.euler['s'] - self.euler['h']) + self.euler['r']
		# also need to check the graph if it matches with the euler equation
		v = self.count_type("vertex")
		f = self.count_type("face")
		e = self.count_type("edge")
		
		graph_match = True
		if self.testing == False:
			assert v == self.euler['v']
			assert f == self.euler['f']
			assert f == len(self.face_cycles)
			assert e == self.euler['e']
		else:
			if v != self.euler['v']:
				print("Number of vertices wrong! Should be ", self.euler['v'], " but in graph ", v)
				graph_match = False
			if f != self.euler['f']:
				print("Number of faces wrong! Should be ", self.euler['f'], " but in graph ", f)
				graph_match = False
			if e != self.euler['e']:
				print("Number of edges wrong! Should be ", self.euler['e'], " but in graph ", e)
				graph_match = False
			if f != len(self.face_loops):
				print("Number of face cycles incorrect! Should be ", self.euler['f'], " but in face cycles ", len(self.face_loops))
				
		if lhs == rhs:
			print("Valid shape")
			valid = True
		else:
			print("Something went wrong: ")
			valid = False
			assert lhs == rhs
		if graph_match:
			print("Graph matches equation")
		return valid
		
	def get_refs(self,line):
		return [el.rstrip(',')          for el in line.replace(","," ").replace("="," ").replace(")"," ").split()          if el.startswith('#')] 
		
	def get_type(self,line):
		print("line: ",line)
		type = re.search("= (.*?)\(",line).group(1)
		return type
	def get_direction(self,line):
		# spilt string into different components
		strings = re.search("\, \( (.*) \) \)", line).group(1)
		strings = strings.split(',')
		# three vectors
		vector = []
		for string in strings:
			vector.append(float(string))
		return vector
		
	def get_radius(self,line):
		# spilt string into different components
		strings = re.search("\, (.*) \)", line).group(1)
		strings = strings.split(',')
		# three vectors
		vector = float(strings[1])
		return vector
		
	def add_nodes(self,refs):
		#first ref is base node
		# check if it exists otherwise there is something wrong with step file or this code!
		assert refs[0] in self.G
		# if refs[0] not in self.G:
			# self.G.add_edge(refs[0])
			# print("Error 1")
		new_level = self.G.nodes[refs[0]]['level'] + 1 # add level for sub node
		if self.max_level < new_level:
			self.max_level = new_level
		#rest are lower nodes
		for node in refs[1:]:
			if node not in self.G:
				#count on level if node added
				try:
					self.level_depth[new_level] += 1
				except:
					self.level_depth[new_level] = 0 #if first on level
				self.G.add_node(node,pos=(self.level_depth[new_level],new_level),level=new_level)
				self.G.add_edge(refs[0],node)
			else:
				self.G.add_edge(refs[0],node)
				
	def add_nodes_2(self,refs):
		#first ref is base node
		# check if it exists otherwise there is something wrong with step file or this code!
		assert refs[0] in self.AG
		# if refs[0] not in self.G:
			# self.G.add_edge(refs[0])
			# print("Error 1")
		#rest are lower nodes
		for node in refs[1:]:
			if node not in self.AG:
				self.AG.add_node(node)
				self.AG.add_edge(refs[0],node)
			else:
				self.AG.add_edge(refs[0],node)
		
	def graph_check(G):
			v # vertices
			e # edges
			f # faces
			s # shells
			h # rings
		
	def load_step(self,step_filename):
		self.filename = os.path.splitext(step_filename)[0]
		self.shell_lines = []
		
		FR = FileReader(step_filename)
		
		# get shells
		shell_lines = []
		shell_lines.extend(FR.find_types('closed_shell'))
		shell_lines.extend(FR.find_types('open_shell'))
		#
		# Number of shells
		shell_count = len(shell_lines)
		print("This model contains ",shell_count," shells.")
		
		#start graph
		self.G = nx.Graph()
		self.level_depth = {}
		self.max_level = 0
		self.level_depth[0] = 0
		self.G.add_node('Root',pos=(0,self.level_depth[0]),level=0,type='Root')
		
		for line in shell_lines:
			shell_refs = self.get_refs(line)
			#for shell add to root node
			root_refs = ['Root',shell_refs[0]]
			self.add_nodes(root_refs)
			self.G.nodes[shell_refs[0]]['type'] = self.get_type(line)
			self.add_nodes(shell_refs)
			#get subnodes
			def sub_add(subnodes):
				for subnode in subnodes:
					sub_lines = FR.find_node(subnode)
					for line in sub_lines:
						refs = self.get_refs(line)
						self.G.nodes[refs[0]]['type'] = self.get_type(line)
						self.add_nodes(refs)
						sub_add(refs[1:])
			sub_add(shell_refs[1:])
			
		
		#Really we want two graphs, the data structure graph and the attributed graph
		#print(self.G.nodes)
		
		#print(nx.get_node_attributes(self.G,'type'))
		# add colours to different levels
		
		#print("Number of nodes ", nx.number_of_nodes(self.G))
	
	def ref_to_int(self,ref):
		return int(ref[1:])
		
	def add_face(self,ref,face_type):
		face_for_adding = self.ref_to_int(ref)
		if self.add_attributes == False:
			self.EG.add_node(('face',face_for_adding),type='face')
		else:
			assert 1 == 0, "need a way of adding attributes"
			
	def add_vertex(self,ref_1):
		ref_1 = self.ref_to_int(ref_1)
		if self.EG.has_node(('vertex',ref_1)):
			print("vertex ",ref_1," already in graph")
		else:
			self.EG.add_node(('vertex',ref_1))
			self.euler['v'] += 1
	
	def add_edge(self,f_ref_1,f_ref_2,v_ref_1,v_ref_2):
		f_ref_1 = self.ref_to_int(f_ref_1)
		f_ref_2 = self.ref_to_int(f_ref_2)
		v_ref_1 = self.ref_to_int(v_ref_1)
		v_ref_2 = self.ref_to_int(v_ref_2)
		face_key = self.EG.add_edge(('face',f_ref_1),('face',f_ref_2),type="face")
		vertex_key = self.EG.add_edge(('vertex',v_ref_1),('vertex',v_ref_2),type="vertex")
		
	
		
	def get_euler_graph(self,step_filename):
		
		self.add_attributes = False
		
		FR = FileReader(step_filename)
		EB = Edge_Bank()
		#step 1  CLOSED_SHELL(face1,...,facek), OPEN_SHELL(face_k+1,...,face_n)
		shell_lines = []
		shell_lines.extend(FR.find_types('closed_shell'))
		shell_lines.extend(FR.find_types('open_shell'))
		self.EG = nx.MultiGraph()
		
		#step 2a
		# ADVANCED_FACE( (bound_1,...,bound_m), surface)
		for line in shell_lines:
# 			print("Shell line:",line) # this is "Closed shell" or Open Shell
			self.euler['s'] += 1
			refs = self.get_refs(line)
			#self.AG.add_node(refs[0],type=self.get_type(line)) #dont need shell on graph
			#self.add_nodes_2(refs)
			for ref in refs[1:]: #these are advance_faces refs
				af_lines = FR.find_node(ref)
				for af_line in af_lines:  #faces
# 					print("af_line:",af_line) 
					af_refs = self.get_refs(af_line)
					self.add_face(ref,self.get_type(af_line))
					self.euler['f'] += 1
# 					self.AG.add_node(ref,type=self.get_type(af_line))
					
					for af_ref in af_refs[1:]:
						# filter bounds and surfaces
						#step 2b
						BorS_lines = FR.find_node(af_ref)
						for BorS_line in BorS_lines:
							#print("BorS_line:",BorS_line)
							BorS = self.get_type(BorS_line)
							#step 2b bound, FACE_OUTER_BOUND(edge_loop) or FACE_BOUND(edge_loop)
							if BorS in ['FACE_BOUND','FACE_OUTER_BOUND']:
								#find edge_loop EDGE_LOOP(edge1,...,edgeu)
								B_refs = self.get_refs(BorS_line)
								for B_ref in B_refs[1:]:
									B_lines = FR.find_node(B_ref)
									#for each edge find ORIENTED_EDGE(E)
									for B_line in B_lines:
										OE_refs = self.get_refs(B_line)
										for OE_ref in OE_refs[1:]:
											OE_lines = FR.find_node(OE_ref)
											for OE_line in OE_lines:
												E_ref = self.get_refs(OE_line)
												prev_E, prev_ref = EB.add_edge(E_ref[1],ref) # This line only adds a fce edge if it appears twice, which it should
												if prev_E == True:
# 													self.AG.add_edge(prev_ref,ref) # add edge attributes later
													# add this later with vertex edge as well self.add_edge_face(prev_ref,ref) # only adding face edge, needs vertex edge adding too
													EC_refs = self.get_refs(OE_line)
													print("OE_line",OE_line)
# 													self.AG.edges[prev_ref,ref]['REF'] = EC_refs[1]
													for EC_ref in EC_refs[1:]:
														EC_lines = FR.find_node(EC_ref)
														for EC_line in EC_lines: ##EDGE_CURVE( '', x, y, z, .T. ); x,y vertex points, z is curve info
															VP_refs = self.get_refs(EC_line)
															VP_ref_x = VP_refs[1]
															VP_ref_y = VP_refs[2]
															VP_ref_curve = VP_refs[3] #B_SPLINE_CURVE_WITH_KNOTS
															print("VP_ref_x",VP_ref_x)
															# make vertex node, if it doesn't already exist
															self.add_vertex(VP_ref_x)
															print("VP_ref_y",VP_ref_y)
															self.add_vertex(VP_ref_y)
															self.add_edge(prev_ref,ref,VP_ref_x,VP_ref_y)
															self.euler['e'] += 1
															
															# the third one needs dealing with
# 															VP_x = RF.find_node(VP_ref_x)
# 															for VP_ref in VP_refs[1:]: #
# 																VP_lines = FR.find_node(VP_ref)
# 																print("VP_lines",VP_lines)
# 													
# 							else:  # find out what kind of surface each of the advance_faces is
# 								ST = self.get_type(BorS_line)
# 								print(ST)
# 								ST_refs = self.get_refs(BorS_line)
# 								# can be the following SURFACE_TYPE
# 								
# 								if ST in ['PLANE']: # PLANE(direction)
# 									self.AG.nodes[ref]['SURFACE_TYPE'] =  ST
# 									for ST_ref in ST_refs[1:]:
# 										AX2_lines = FR.find_node(ST_ref)  # AXIS2_PLACEMENT_3D (origin,z ,x)
# 										for AX2_line in AX2_lines:
# 											AX2_refs = self.get_refs(AX2_line)
# 											# AX2_refs[1] = origin, not required for some reason
# 											z_lines = FR.find_node(AX2_refs[2]) # DIRECTION_Z
# 											x_lines = FR.find_node(AX2_refs[3]) # DIRECTION_X
# 											for z_line in z_lines:
# 												self.AG.nodes[ref]['DIRECTION_Z'] = self.get_direction(z_line)
# 											for x_line in x_lines:
# 												self.AG.nodes[ref]['DIRECTION_X'] = self.get_direction(x_line)
# 												
# 								elif ST in ['CYLINDRICAL_SURFACE']: # CYLINDRICAL_SURFACE(direction,radius)  
# 								# currently not saving radius
# 									self.AG.nodes[ref]['SURFACE_TYPE'] =  ST
# 									for ST_ref in ST_refs[1:]:
# 										AX2_lines = FR.find_node(ST_ref)  # AXIS2_PLACEMENT_3D (origin,z ,x)
# 										for AX2_line in AX2_lines:
# 											AX2_refs = self.get_refs(AX2_line)
# 											# AX2_refs[1] = origin, not required for some reason
# 											z_lines = FR.find_node(AX2_refs[2]) # DIRECTION_Z
# 											x_lines = FR.find_node(AX2_refs[3]) # DIRECTION_X
# 											for z_line in z_lines:
# 												self.AG.nodes[ref]['DIRECTION_Z'] = self.get_direction(z_line)
# 											for x_line in x_lines:
# 												self.AG.nodes[ref]['DIRECTION_X'] = self.get_direction(x_line)
# 									
# 								elif ST in ['CONICAL_SURFACE']: # CONICAL_SURFACE(direction,radius, semi angle)
# 								# currently not saving radius or semi angle
# 									self.AG.nodes[ref]['SURFACE_TYPE'] =  ST
# 									for ST_ref in ST_refs[1:]:
# 										AX2_lines = FR.find_node(ST_ref)  # AXIS2_PLACEMENT_3D (origin,z ,x)
# 										for AX2_line in AX2_lines:
# 											AX2_refs = self.get_refs(AX2_line)
# 											# AX2_refs[1] = origin, not required for some reason
# 											z_lines = FR.find_node(AX2_refs[2]) # DIRECTION_Z
# 											x_lines = FR.find_node(AX2_refs[3]) # DIRECTION_X
# 											for z_line in z_lines:
# 												self.AG.nodes[ref]['DIRECTION_Z'] = self.get_direction(z_line)
# 											for x_line in x_lines:
# 												self.AG.nodes[ref]['DIRECTION_X'] = self.get_direction(x_line)
# 												
# 								elif ST in ['SPHERICAL_SURFACE']: # SPHERICAL_SURFACE(direction, radius, semi angle)
# 								# currently not saving radius or semi angle
# 									self.AG.nodes[ref]['SURFACE_TYPE'] =  ST
# 									for ST_ref in ST_refs[1:]:
# 										AX2_lines = FR.find_node(ST_ref)  # AXIS2_PLACEMENT_3D (origin,z ,x)
# 										for AX2_line in AX2_lines:
# 											AX2_refs = self.get_refs(AX2_line)
# 											# AX2_refs[1] = origin, not required for some reason
# 											z_lines = FR.find_node(AX2_refs[2]) # DIRECTION_Z
# 											x_lines = FR.find_node(AX2_refs[3]) # DIRECTION_X
# 											for z_line in z_lines:
# 												self.AG.nodes[ref]['DIRECTION_Z'] = self.get_direction(z_line)
# 											for x_line in x_lines:
# 												self.AG.nodes[ref]['DIRECTION_X'] = self.get_direction(x_line)
# 												
# 								elif ST in ['TOROIDAL_SURFACE']: # TOROIDAL_SURFACE(direction, major_radius, minor_radius)
# 								# currently not saving major or minor radius
# 									self.AG.nodes[ref]['SURFACE_TYPE'] =  ST
# 									for ST_ref in ST_refs[1:]:
# 										AX2_lines = FR.find_node(ST_ref)  # AXIS2_PLACEMENT_3D (origin,z ,x)
# 										for AX2_line in AX2_lines:
# 											AX2_refs = self.get_refs(AX2_line)
# 											# AX2_refs[1] = origin, not required for some reason
# 											z_lines = FR.find_node(AX2_refs[2]) # DIRECTION_Z
# 											x_lines = FR.find_node(AX2_refs[3]) # DIRECTION_X
# 											for z_line in z_lines:
# 												self.AG.nodes[ref]['DIRECTION_Z'] = self.get_direction(z_line)
# 											for x_line in x_lines:
# 												self.AG.nodes[ref]['DIRECTION_X'] = self.get_direction(x_line)
# 								else:
# 									self.AG.nodes[ref]['SURFACE_TYPE'] =  'BOUNDED_SURFACE' # BOUNDED_SURFACE() these are a bit more complex!
# 									# print(BorS_line)
# 									# currently not saving anything
# 									
# 					#print(af_line) # face is 'ref'
# 		# complete nodes before doing edge stuff
# 		#print(self.AG.edges.data() )
# 		for edge in self.AG.edges:
# 			# get EDGE_CURVE
# 			EC_ref = self.AG.edges[edge]['REF']
# 			EC_lines = FR.find_node(EC_ref)
# 			for EC_line in EC_lines:
# 				CT_refs = self.get_refs(EC_line)  ## CURVE TYPE
# 				CT_lines = FR.find_node(CT_refs[3])
# 				for CT_line in CT_lines:
# 					# can be one of three types
# 					CT_type = self.get_type(CT_line)
# 					if CT_type in ['LINE']: ## LINE(point,vector)
# 						self.AG.edges[edge]['CURVE_TYPE'] = CT_type
# 						print('THIS NEEDS SORTING') # need to find an example of this!
# 					elif CT_type in ['CIRCLE']: # CIRCLE (direction, radius)
# 						self.AG.edges[edge]['CURVE_TYPE'] = CT_type
# 						try:
# 							#print(self.AG.nodes[edge[0]])
# 							#print(self.AG.nodes[edge[1]])
# 							z1 = self.AG.nodes[edge[0]]['DIRECTION_Z']
# 							x1 = self.AG.nodes[edge[0]]['DIRECTION_X']
# 							z2 = self.AG.nodes[edge[1]]['DIRECTION_Z']
# 							x2 = self.AG.nodes[edge[1]]['DIRECTION_X']
# 							rel_z = np.arccos(np.dot(z1,z2))
# 							rel_x = np.arccos(np.dot(x1,x2))
# 							self.AG.edges[edge]['RELATIVE_DIR_Z'] = rel_z
# 							self.AG.edges[edge]['RELATIVE_DIR_X'] = rel_x
# 							self.AG.edges[edge]['LENGTH'] = self.get_radius(CT_line)
# 						except:
# 							self.AG.edges[edge]['RELATIVE_DIR_Z'] = 0
# 							self.AG.edges[edge]['RELATIVE_DIR_X'] = 0
# 							self.AG.edges[edge]['LENGTH'] = 0
# 					elif CT_type in ['B_SPLINE_CURVE_WITH_KNOTS']: #B_SPLINE_CURVE_WITH_KNOTS()
# 						try:
# 							#print(self.AG.nodes[edge[0]])
# 							#print(self.AG.nodes[edge[1]])
# 							z1 = self.AG.nodes[edge[0]]['DIRECTION_Z']
# 							x1 = self.AG.nodes[edge[0]]['DIRECTION_X']
# 							z2 = self.AG.nodes[edge[1]]['DIRECTION_Z']
# 							x2 = self.AG.nodes[edge[1]]['DIRECTION_X']
# 							rel_z = np.arccos(np.dot(z1,z2))
# 							rel_x = np.arccos(np.dot(x1,x2))
# 							self.AG.edges[edge]['RELATIVE_DIR_Z'] = rel_z
# 							self.AG.edges[edge]['RELATIVE_DIR_X'] = rel_x
# 							self.AG.edges[edge]['LENGTH'] = 0
# 						except:
# 							self.AG.edges[edge]['RELATIVE_DIR_Z'] = 0
# 							self.AG.edges[edge]['RELATIVE_DIR_X'] = 0
# 							self.AG.edges[edge]['LENGTH'] = 0
# 						#print(self.AG.edges[edge])
# 					else:
# 						pass
# 						#print(CT_type)
# 			#print(self.AG.edges[edge])
# 			
			
# 		labels=dict((n,d['SURFACE_TYPE']) for n,d in self.AG.nodes(data=True))
		plt.figure(3,figsize=self.plot_size)
		nx.draw(self.EG,node_size=1000,node_color='skyblue', pos=nx.spring_layout(self.EG),with_labels=True)#labels=labels)#
		plt.show()
		#self.euler_valid()
		
		
	def get_simple_graph(self,step_filename,save_graph=False,draw_graph=True):
		self.filename = os.path.basename(step_filename)
		self.filename = os.path.splitext(self.filename)[0]
		FR = FileReader(step_filename)
		EB = Edge_Bank()
		#step 1  CLOSED_SHELL(face1,...,facek), OPEN_SHELL(face_k+1,...,face_n)
		shell_lines = []
		shell_lines.extend(FR.find_types('closed_shell'))
		shell_lines.extend(FR.find_types('open_shell'))
		self.AG = nx.MultiGraph()
		print("len(shell_lines)",len(shell_lines))
		#step 2a
		# ADVANCED_FACE( (bound_1,...,bound_m), surface)
		for line in shell_lines:
			#print("line",line)
			refs = self.get_refs(line)
			#print("refs: ",refs)
			#self.AG.add_node(refs[0],type=self.get_type(line)) #dont need shell on graph
			#self.add_nodes_2(refs)
			for ref in refs[1:]: #these are advance_faces refs
				#print("Ref:",ref)
				af_lines = FR.find_node(ref)
				#print("af_lines:",len(af_lines))
				for af_line in af_lines:
					af_refs = self.get_refs(af_line)
					#print("af_refs:",af_refs)
					#print("af_line:",af_line)
					self.AG.add_node(ref,type=self.get_type(af_line))
					
					for af_ref in af_refs[1:]:
						# fiture bounds and edges
						#step 2b
						BorE_lines = FR.find_node(af_ref)
						for BorE_line in BorE_lines:
							BorE = self.get_type(BorE_line)
							#step 2b bound, FACE_OUTER_BOUND(edge_loop) or FACE_BOUND(edge_loop)
							if BorE in ['FACE_BOUND','FACE_OUTER_BOUND']:
								#find edge_loop EDGE_LOOP(edge1,...,edgeu)
								B_refs = self.get_refs(BorE_line)
								for B_ref in B_refs[1:]:
									B_lines = FR.find_node(B_ref)
									#for each edge find ORIENTED_EDGE(E)
									for B_line in B_lines:
										OE_refs = self.get_refs(B_line)
										for OE_ref in OE_refs[1:]:
											OE_lines = FR.find_node(OE_ref)
											for OE_line in OE_lines:
												E_ref = self.get_refs(OE_line)
												prev_E, prev_ref = EB.add_edge(E_ref[1],ref)
												if prev_E == True:
													# print(self.AG.has_edge(prev_ref,ref))
													self.AG.add_edge(prev_ref,ref) # add edge attributes later
		
		# extract sub graphs
		sub_graphs = True
		multi = False
		if sub_graphs == True:
			multi = True
			sub_graphs = (self.AG.subgraph(c) for c in nx.connected_components(self.AG))
		else:
			
			sub_graphs = self.AG
			
		for i, sg in enumerate(sub_graphs):
			sg = nx.convert_node_labels_to_integers(sg)
			c_str = ''
			if multi == True:
				c_str = str(i) 
			
			if save_graph == True:
				nx.write_adjlist(sg,self.filename + '_' + c_str + '.adjlist')
			
			if draw_graph == True:
				plt.figure(i,figsize=self.plot_size)
				nx.draw(sg,node_size=1000,node_color='skyblue', pos=nx.spring_layout(sg),with_labels=True)
				plt.show()
	
	def create_tree(self):
		#cretae tree in newick format
		self.tree = Tree()
	
	def print_tree(self):
		groups = set(nx.get_node_attributes(self.G,'level').values())
		mapping = dict(zip(sorted(groups),count()))
		nodes = self.G.nodes()
		colours = [mapping[self.G.nodes[n]['level']] for n in nodes]
		pos=nx.get_node_attributes(self.G,'pos')
		# normalise pos so it is centred
		range = max(self.level_depth)
		range = self.level_depth[range]
		pos2 = {}
		for k, v in pos.items():
			row_max = self.level_depth[v[1]]
			try:
				x = range*(v[0]/row_max)
			except: #if row_max = 0!
				x = range/ 2
			pos2[k] = (x, self.max_level - v[1])
		plt.figure(1,figsize=self.plot_size)
		nx.draw(self.G,pos2,with_labels=False,node_color=colours, node_size=80, cmap=plt.cm.tab20)
		plt.show()
		
	def print_tree2(self):
		groups = set(nx.get_node_attributes(self.G,'type').values())
		mapping = dict(zip(sorted(groups),count()))
		print(mapping)
		nodes = self.G.nodes()
		colours = [mapping[self.G.nodes[n]['type']] for n in nodes]
		pos=nx.get_node_attributes(self.G,'pos')
		# normalise pos so it is centred
		range = max(self.level_depth)
		range = self.level_depth[range]
		pos2 = {}
		for k, v in pos.items():
			row_max = self.level_depth[v[1]]
			try:
				x = range*(v[0]/row_max)
			except: #if row_max = 0!
				x = range/ 2
			pos2[k] = (x, self.max_level - v[1])
		plt.figure(2,figsize=self.plot_size)
		nx.draw(self.G,pos2,with_labels=False,node_color=colours, node_size=80, cmap=plt.cm.tab20)
		plt.draw()
		

import matplotlib.image as mpimg
num = '986' #'003'
model = 'C:/Users/prctha/PythonDev/DCS-project-master/examples/models/abc_' + num +'.step'
#img = mpimg.imread('C:/Users/prctha/PythonDev/DCS-project-master/examples/models/abc_' + num +'.png')

#model = 'C:/Users/prctha/PythonDev/DCS-project-master/examples/models/main assy.stp'

#model = "C:/Users/prctha/PythonDev/DCS-project-master/examples/models/on-pallet-pressing-machine-with-conveyor-1.snapshot.2/On Pallet Pressing Machine.STEP"
#img = mpimg.imread("C:/Users/prctha/PythonDev/DCS-project-master/examples/models/on-pallet-pressing-machine-with-conveyor-1.snapshot.2/On pallet pressing01.PNG")



#imgplot = plt.imshow(img)
SP = Step_brep()
SP.get_simple_graph(model,save_graph=False)
#SP.get_euler_graph(model)
SP.load_step(model)
SP.print_tree()
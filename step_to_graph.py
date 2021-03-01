# TH Sept 2020
# Step to Attributed graphs from b-rep modesl
"""
step_to_graph
"""

import re
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import shutil


import matplotlib.image as mpimg
num = '986' #'003'
#model = 'C:/Users/prctha/PythonDev/DCS-project-master/examples/models/abc_' + num +'.step'
#model = 'C:/Users/prctha/PythonDev/DCS-project-master/examples/models/main assy.stp'
#model = "C:/Users/prctha/PythonDev/GitHub/DCS-project/examples/models/Torch Assembly.STEP"


class StepToGraph():
	
	def __init__(self,step_filename):
		self.filename = step_filename
		fileHandler = open(self.filename,"r")
		self.listOfLines = fileHandler.readlines()
		fileHandler.close()
		
		self.dup_dict = {} # for tracking duplicate parts
		
		print("file_loaded")
		self.G = nx.MultiDiGraph()
		self.H = nx.MultiDiGraph()
		self.F = nx.Graph()
		self.get_nodes()
		print("non: ",self.G.number_of_nodes())
		self.get_edges()
		print("non: ",self.G.number_of_nodes())
		#self.filter_nodes()
		#print("nodes filtered")
		
		#print("part list = ",self.part_list)
		#print("part list obtained")
		self.get_face_adj()
		print("done face_adj")
		#self.print_graph()
		#self.print_parts()
		print("Number of edges:")
		print(self.H.number_of_edges())
		#self.print_fadj_only()
		
		
		
	
	def print_graph(self):
		#groups = set(nx.get_node_attributes(self.G,'Geometric').values())
		
		groups = set(nx.get_node_attributes(self.G,'type').values())
		tests = {'CONICAL_SURFACE', 'CIRCLE', 'TOROIDAL_SURFACE', 'BOUNDED_SURFACE','FILL_AREA_STYLE', 'PRODUCT_DEFINITION_SHAPE', 'APPLICATION_PROTOCOL_DEFINITION', 'CONICAL_SURFACE', 'PRODUCT', 'UNCERTAINTY_MEASURE_WITH_UNIT', 'FACE_OUTER_BOUND', 'CARTESIAN_POINT', 'COLOUR_RGB', 'NAMED_UNIT', 'PRODUCT_DEFINITION_CONTEXT', 'CIRCLE', 'B_SPLINE_CURVE_WITH_KNOTS', 'ADVANCED_BREP_SHAPE_REPRESENTATION', 'MANIFOLD_SOLID_BREP', 'EDGE_CURVE', 'CONVERSION_BASED_UNIT', 'FACE_BOUND', 'PLANE', 'ORIENTED_EDGE', 'LENGTH_MEASURE_WITH_UNIT', 'TOROIDAL_SURFACE', 'AXIS2_PLACEMENT_3D', 'EDGE_LOOP', 'BOUNDED_SURFACE', 'SHAPE_DEFINITION_REPRESENTATION', 'SURFACE_STYLE_FILL_AREA', 'PRESENTATION_STYLE_ASSIGNMENT', 'DIRECTION', 'ADVANCED_FACE', 'SURFACE_STYLE_USAGE', 'PRODUCT_CONTEXT', 'CLOSED_SHELL', 'SURFACE_SIDE_STYLE', 'PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE', 'PRODUCT_CATEGORY_RELATIONSHIP', 'DIMENSIONAL_EXPONENTS', 'PRODUCT_DEFINITION', 'VERTEX_POINT', 'PRODUCT_RELATED_PRODUCT_CATEGORY', 'CYLINDRICAL_SURFACE', 'STYLED_ITEM', 'APPLICATION_CONTEXT', 'PRODUCT_CATEGORY', 'GEOMETRIC_REPRESENTATION_CONTEXT', 'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION', 'LENGTH_UNIT', 'FILL_AREA_STYLE_COLOUR'}
		for t in groups:
			if t in tests:
				tests.remove(t)
		print("types: ", groups)
		print("extras: ",tests)
		mapping = dict(zip(sorted(groups),count()))
		nodes = self.G.nodes()
		colours = [mapping[self.G.nodes[n]['type']] for n in nodes]
		pos = graphviz_layout(self.G, prog="dot")
		self.plot_size = (100,100)
		plt.figure(2,figsize=self.plot_size)
		labels = {}
		for node in nodes:
			labels[node]  = str(node) + " , " + str(self.G.nodes[node]['type'])
		nx.draw(self.G,pos=pos,labels=labels,node_color=colours)
		plt.show()
		plt.savefig("tree.png")
		
	def print_parts(self):
		#groups = set(nx.get_node_attributes(self.G,'Geometric').values())
		nodes = self.H.nodes()
		#pos = graphviz_layout(self.H, prog="dot")
		self.plot_size = (40,40)
		plt.figure(2,figsize=self.plot_size)
		labels = {}
		#for node in nodes:
		#	labels[node]  = str(node) + " , " + str(self.H.nodes[node]['label'])
		nx.draw(self.H)#,pos=pos,labels=labels)
		plt.show()
		plt.savefig("tre2e.png")
		
	def print_fadj(self):
		nodes = self.H.nodes()
		groups = set(nx.get_node_attributes(self.H,'type').values())
		#print("types: ", groups)
		mapping = dict(zip(sorted(groups),count()))
		colours = [mapping[self.H.nodes[n]['type']] for n in nodes]
		self.plot_size = (100,100)
		plt.figure(3,figsize=self.plot_size)
		nx.draw(self.H,node_color=colours)
		plt.show()
		plt.savefig("tree2.png")
		
	def print_fadj_only(self):
		nodes = [x for x,y in self.H.nodes(data=True) if y['type']=='face']
		sub_h = self.H.subgraph(nodes)
		
		# make undirected copy
		u_sub_h = sub_h.to_undirected()
		sub_H = [u_sub_h.subgraph(c).copy() for c in nx.connected_components(u_sub_h)]
		#print(S)
		
		#sub_H = nx.connected_component_subgraphs(u_sub_h)
		
		self.plot_size = (6,6)
		for i, sg in enumerate(sub_H):
			plt.figure(i,figsize=self.plot_size)
			nx.draw(sg)
			plt.show()
	
	def export_graph(self,filename):
		# .gz
		nx.write_gpickle(self.H,filename)
		print("graph saved")
	
	def get_type(self,line):
		#print("line: ",line)
		type = re.search("=(.*?)\(",line).group(1)
		if type == "":
			type = re.search("=\((.*?)\(",line).group(1)
		return type
	
	def line_filter1(self,line_type):
		useful = False
		if line_type in ["NEXT_ASSEMBLY_USAGE_OCCURRENCE", 'PRODUCT_DEFINITION', 'PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE','PRODUCT','PRODUCT_DEFINITION_SHAPE','SHAPE_DEFINITION_REPRESENTATION','SHAPE_REPRESENTATION']:#'PRODUCT_CONTEXT'
			useful = True
		if line_type in ['EDGE_CURVE','ORIENTED_EDGE','EDGE_LOOP','FACE_BOUND','FACE_OUTER_BOUND','ADVANCED_FACE','CLOSED_SHELL','MANIFOLD_SOLID_BREP','ADVANCED_BREP_SHAPE_REPRESENTATION','SHAPE_REPRESENTATION_RELATIONSHIP','SHAPE_REPRESENTATION']:#''ADVANCED_FACE']:
			useful = True
		return useful
	
	def line_filter2(self,line_type):
		useful = False
		if line_type in []:
			useful = True
		return useful
	
	def get_nodes(self):
		# This function goes through every line and gets the node ID, edges added later
		
		i = 0
		
		cont_line = ""
		new_line = True
		for line in self.listOfLines:
# 			i += 1
# 			if i>20:
# 				break
			raw_line = "".join(line.split())
			#print(raw_line)
			if new_line == False:
				raw_line = cont_line + raw_line
				#print("new_line", raw_line)
				cont_line = ""
			if raw_line == "":
				continue
			if raw_line[0] != "#":
#				if cont_line != "": # something saved
#					raw_line = cont_line + raw_line
#					cont_line = ""
#				else:
				#print("not a proper line")
				continue
			#check line ends with ";", if not line carries over
			# new item
			if raw_line[-1] == ";":
				#extract data
				#print(raw_line)
				index = re.search("#(.*)=", raw_line)
				try:
					line_inx = index.group()
					line_inx = line_inx[1:-1]
				except:
					print("cont_line: ",cont_line)
					print("broken line: ", raw_line)
				#print("line inx:", line_inx)
				line_type = self.get_type(raw_line)
				#print("line type:", line_type)
				if self.line_filter1(line_type):
					#print(raw_line)
					if line_type == "PRODUCT":
						self.add_part(line_inx,raw_line)
					self.G.add_node(line_inx,type=line_type)
					if line_type == "ADVANCED_FACE":
						self.add_face(line_inx)
				cont_line = "" # reset line spill over
				new_line = True
				continue
			else:
				# line started but not finished
				cont_line = cont_line + raw_line
				new_line = False
			
	def get_edges(self):
		# This function goes through every line and gets adds the edges
		
		i = 0
		
		parent_list = []
		child_list = []
		
		cont_line = ""
		new_line =True
		for line in self.listOfLines:
# 			i += 1
# 			if i>20:
# 				break
			raw_line = "".join(line.split())
			#print(raw_line)
			if new_line == False:
				raw_line = cont_line + raw_line
				cont_line = ""
			if raw_line == "":
				continue
			if raw_line[0] != "#":
				continue
			#check line ends with ";", if not line carries over
			# new item
			if raw_line[-1] == ";":
				#extract data
				links = []
				#print(raw_line)
				indices = re.split("#",raw_line)
				# remove first 2 items, '' and line number
				indices.pop(0)
				indices.pop(0)
				index = re.search("#(.*)=", raw_line)
				try:
					line_inx = index.group()
				except:
					print("broken line: ", raw_line)
				line_inx = line_inx[1:-1]
				if self.G.has_node(line_inx):
					
					#print("index: ",index)
					line_type = self.get_type(raw_line)
					i = 0
					for index in indices:
						link = re.search('[0-9]+', index).group()
						if self.G.has_node(link):
							if line_type == 'NEXT_ASSEMBLY_USAGE_OCCURRENCE':
								if i == 0:
									parent_list.append(link)
								elif i == 1:
									child_list.append(link)
							self.G.add_edge(line_inx,link)
							links.append(link)
						i += 1
				#print("indices: ", links)
				
				cont_line = "" # reset line spill over
				continue
			else:
				# line started but not finished
				cont_line = raw_line
				new_line = False
		
		#print("parent:",parent_list)
		#print("child:",child_list)
		root_ref = set(parent_list) - set(child_list)
		#print("root:",root_ref)
		
		
		for i, p in enumerate(parent_list):
			#print(i)
			self.add_part_edge(parent_list[i],child_list[i])
		
		self.get_part_list()
		#print("self.part_list")
		#print(self.part_list)
		self.join_faces()
		self.link_fadjs()
		
				
	def add_part(self,idx,line):
		part_name = line.split("'")
		#print("line:",part_name[1])
		#print("idx:",idx)
		self.H.add_node(idx,type="assembly",label=part_name[1])
		print("part:",self.H.nodes[idx]["type"])
        
	def add_part_edge(self,parent_idx,child_idx):
		# find connected part name
		p_idx = self.get_part(parent_idx)
		c_idx = self.get_part(child_idx)
		
		# need to check for duplicates
		#print(self.H.has_edge(p_idx,c_idx))
		if self.H.has_edge(p_idx,c_idx):
			#needs duplicating
			#print(self.H.nodes[c_idx])
			if c_idx in self.dup_dict:
				self.dup_dict[c_idx] += 1
				c_idx_new = c_idx + "_" + str(self.dup_dict[c_idx])
			else:
				self.dup_dict[c_idx] = 1
				c_idx_new = c_idx + "_1" 
			self.H.add_node(c_idx_new,type='face')
			self.H.nodes[c_idx_new]['label'] = self.H.nodes[c_idx]['label']
			c_idx = c_idx_new
		key = self.H.add_edge(p_idx,c_idx)
		self.H.edges[p_idx,c_idx, key]['type'] = 'assembly'
		
	def add_face(self,idx):
		self.F.add_node(idx)
		self.H.add_node(idx,type='face')
		
	def join_faces(self):
		connected = [0, 0]
		for n, t in self.G.nodes('type'):
			edge_pair = []
			if t == "EDGE_CURVE":
				connected[1] += 1
				if self.G.in_degree(n) == 2:
					oe_nodes = list(self.G.predecessors(n)) #these are oriented_edges
					for oe_node in oe_nodes:
						el_nodes = list(self.G.predecessors(oe_node)) #these are edge loops
						for el_node in el_nodes:
							fb_nodes = list(self.G.predecessors(el_node))#these are face_outer_bound or face_bound nodes
							for fb_node in fb_nodes:
								assert self.G.in_degree(fb_node) == 1, "Error join_faces_2"
								af_nodes = list(self.G.predecessors(fb_node))
								edge_pair.append(af_nodes[0])
				else:
					print("error join_faces")
				#if len(edge_pair) != 2:
				#	connected[0] += 1
					#print(n,t)
					#print(edge_pair)
					#print(self.G.predecessors(n))
				assert len(edge_pair) == 2, "Error join_faces 3" 
				
				self.F.add_edge(edge_pair[0],edge_pair[1])
				# needs both directions
				key = self.H.add_edge(edge_pair[0],edge_pair[1])
				self.H.edges[edge_pair[0],edge_pair[1], key]['type'] = 'face'
				key = self.H.add_edge(edge_pair[1],edge_pair[0])
				self.H.edges[edge_pair[1],edge_pair[0], key]['type'] = 'face'
		print("There are ", connected[0], " edge_curves out of ", connected[1], " are not connected for some reason")
	
	def link_fadjs(self):
		# link part to "closed shell"
		# look for SHAPE_REPRESENTATION nodes
		def find_product(n,prev_n=''):
			if self.G.nodes[n]['type'] == 'PRODUCT':
				return n, 'PRODUCT'
			elif self.G.nodes[n]['type'] == 'CLOSED_SHELL':
				return n, 'CLOSED_SHELL'
			else:
				if self.G.out_degree(n) == 1:
					m = list(self.G.successors(n))
					return find_product(m[0])
				elif self.G.out_degree(n) == 2:
					m = list(self.G.successors(n))
					if prev_n in m:
						m.remove(prev_n)
						return find_product(m[0])
					else:
						print("errorroro")
				else:
					print("n:", n)
					print("ERROOOROROROROR")
					
		
		for n, t in self.G.nodes('type'):
			if t == "SHAPE_REPRESENTATION":
				#look for incoming nodes
				if self.G.in_degree(n) == 2: # if number of in_nodes == 2 then a part otherwise an assembly
					in_nodes = list(self.G.predecessors(n))
					#print("in_nodes for ",n, " are: ", in_nodes)
					node_product = []
					node_closed_shell = []
					for m in in_nodes:
						node, typ = find_product(m,prev_n=n)
						if typ == "CLOSED_SHELL":
							node_closed_shell = node
						elif typ == "PRODUCT":
							node_product = node
					af_nodes = list(self.G.successors(node_closed_shell))
					
					#self.H.add_node(node_closed_shell,label='shell')
					for af_node in af_nodes:
						# check af_node in dup list
						if node_product in self.dup_dict:
							#print(node_product in self.dup_dict)
							for i in range(1,self.dup_dict[node_product]+1):
								node_product_new = node_product + "_" + str(i)
								key = self.H.add_edge(af_node,node_product_new)  # direction swapped
								self.H.edges[af_node,node_product_new,key]['type'] = "link"
						key = self.H.add_edge(af_node,node_product)  # direction swapped
						self.H.edges[af_node,node_product,key]['type'] = "link"
	
	def get_part(self,idx):
		for n in nx.neighbors(self.G,idx):
			for m in nx.neighbors(self.G,n):
				if self.G.has_node(m):
					return m
				print("ERROR!")
				return 0
			
			
	def get_part_list(self):
		# makes a list of parts the are the leaf nodes of self.H
# 		for x in self.H.nodes():
# 			if self.H.nodes[x]['type'] == 'assembly':
# 				print(x)
# 				print(self.H.nodes[x]['label'])
# 				print("self.H.out_degree(x)")
# 				print(self.H.out_degree(x))
# 				print("self.H.in_degree(x)")
# 				print(self.H.in_degree(x))
		self.part_list = [ x for x in self.H.nodes() if self.H.out_degree(x) == 0 and self.H.in_degree(x)==1]
		for part in self.part_list:
			self.H.nodes[part]['type'] = "part"
			
	def get_face_adj(self):
		# for each part, so each leaf in self.H
		#for part in self.part_list:
			# need to get face adj for each part
		#	print("part: ", part)
			
		# find 'ADVANCED_FACE's
		for (n, t) in self.G.nodes('type'):
			if t == 'SHAPE_REPRESENTATION':
				pass
				#print(n ,t)
		
	
	def filter_nodes(self):
		# Filters out the parts of the graph we arent interested in
		
		# main node is type "ADVANCED_BREP_SHAPE_REPRESENTATION"
		nx.set_node_attributes(self.G, "False" , "Geometric")
		
		def tag_connected(node):
			for n in nx.neighbors(self.G,node):
				self.G.nodes[n]['Geometric'] = "True"
				tag_connected(n)
				
		root_node = []
		for (n, t) in self.G.nodes('type'):
			if t == "ADVANCED_BREP_SHAPE_REPRESENTATION":
				root_node.append(n)
				
		print("Root node: ", root_node)
		self.G.nodes[root_node[0]]['Geometric'] = "True"
		tag_connected(root_node[0])
		
		for (n, d) in self.G.nodes('type'):
			if d in ['PRODUCT_DEFINITION','PRODUCT_DEFINITION_FORMATION','PRODUCT' ]:
				print("geometric:",self.G.nodes[n]['Geometric'])
				self.G.nodes[n]['Geometric'] = "Product"
				print("prduct nodes:",n)


def load_step(step_file,remove_tmp=True):
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
	T = StepToGraph(tmp_file)
	os.remove(tmp_file)
	#print("T.H",type(T.H))
	#T.print_graph()
	#T.print_fadj()
	T.export_graph(os.path.join('./tmp/','test.gz'))
	return T.H


# folderin = "E:/ABC-Dataset/Step/step_abc"
# folderout = "E:/ABC-Dataset/gz_files"

# import os
# II = 0
# for filename in os.listdir(folderin):
	# II += 1
	# print("processing number:" , II)
	# try:
		# T = StepToGraph(os.path.join(folderin,filename))
		# fileout = os.path.splitext(filename)[0] + ".gz"
		# T.export_graph(os.path.join(folderout,fileout))
	# except:
		# print("error with file:",filename)
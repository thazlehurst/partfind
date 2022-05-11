# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:55:25 2021

@author: prctha
"""


from step2graph import StepToGraph
import os
from tqdm import tqdm


def create_dataset(dataset_folder,output_file, filter_list = None):
    
    import pickle
    if not isinstance(dataset_folder, list):
        pk_path = os.path.join(dataset_folder,output_file)
        
        
        dataset = {}
        
        i = 0 
        
        total = 0
        for root, subdirs, files in os.walk(dataset_folder):
            for file in files:
                filename, file_extension = os.path.splitext(file)
                if file_extension in [".stp",".step",".STEP",".STP"]:
                    total += 1
        print("Total step files:", total)
        
        start_point = 0
        good_count = 0
        pbar = tqdm(total=total)
        problem_files = []
        for root, subdirs, files in os.walk(dataset_folder):
            path = root.split(os.sep)
            for file in files:
                
                filename, file_extension = os.path.splitext(file)
                if file_extension in [".stp",".step",".STEP",".STP"]:
                    i += 1
                    pbar.update(1)
                    if filter_list != None:
                        if filename not in filter_list:
                            print("Not in filter_list: ",filename)
                            continue
                    #print(path[-2],file)
                    filepath = os.path.join(root,file)
                    try:
                        if i < start_point:
                            continue
                        print(filename)
                        
                        if filename in ['00000360_9ee540a56d79451e8026e1e6_step_002',    '0004376_0161a8b664ac4b21aaf2f0c4_step_000']:
                            raise ValueError('Dud')
                        try:    
                            with open(pk_path,'rb') as f:
                                dataset = pickle.load(f)
                                #print(dataset.keys())
                        except:
                            print("no current pickle file")
                        print("in dataset?", filename in dataset)
                        if filename not in dataset:
                            
                            s2g = StepToGraph(filepath)
                            s2g.compute_faces_surface()
                                
                            
                            dataset[filename] = {"cat":path[-2],"graph_nx":s2g.G}
                            
                            with open(pk_path,'wb') as handle:
                                pickle.dump(dataset,handle, protocol=pickle.HIGHEST_PROTOCOL)
                            good_count += 1
                        
                        #print(i, "/",total)
                    except Exception as inst:
                        print(inst)
                        problem_files.append(filepath)
                        textfile = open(os.path.join(dataset_folder,"problems.txt"), "w")
                        for element in problem_files:
                            textfile.write(element + "\n")
                        textfile.close()
                        print("problem with:", filepath)
        pbar.close()
        print(dataset)
        print("problem files",problem_files)
        print("good count",good_count)
        # save problems
        
        textfile = open(os.path.join(dataset_folder,"problems.txt"), "w")
        for element in problem_files:
            textfile.write(element + "\n")
        textfile.close()
    else:
        # file list
        model_list = dataset_folder
        pbar = tqdm(total=len(model_list))
        pk_path = output_file
        dataset = {}
        for filepath in model_list:
            print()
            print(filepath)
            file_path_name, file_extension = os.path.splitext(filepath)
            print(file_path_name)
            _, filename = os.path.split(file_path_name)
            print(filename)
            try:    
                with open(pk_path,'rb') as f:
                    dataset = pickle.load(f)
                    #print(dataset.keys())
            except:
                print("no current pickle file")
            print()
            print("in dataset?", filename in dataset)
            print()
            if filename not in dataset:
                s2g = StepToGraph(filepath)
                s2g.compute_faces_surface()
                dataset[filename] = {"cat":"None","graph_nx":s2g.G}
                with open(pk_path,'wb') as handle:
                    pickle.dump(dataset,handle, protocol=pickle.HIGHEST_PROTOCOL)
            pbar.update(1)
        pbar.close()
        problem_files = []
        good_count = 0
 
    
def ex_bool(x):
    if x == 'False':
        return False
    else:
        return True
    
def read_triples(file):
    used_list = []
    useful_triples = 0
    import csv
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 1:
                continue
            userid = row[0]
            model1 = row[2]
            model2 = row[3]
            model3 = row[4]
            sim1 = ex_bool(row[5])
            sim2 = ex_bool(row[6])
            sim3 = ex_bool(row[7])
            
            if sim1 + sim2 + sim3 == 2:
                useful_triples += 1
                if model1 not in used_list:
                    used_list.append(model1)
                if model2 not in used_list:
                    used_list.append(model2)
                if model3 not in used_list:
                    used_list.append(model3)
    print("Total models:",len(used_list))
    print("Useful triples:",useful_triples)
    return used_list
    
if __name__ == "__main__":
    #dataset_folder = "C:\\Users\\prctha\\PythonDev\\ABC_Data"
    #output_file = "ABC_nx.pickle"
    filter_list = read_triples("exp1_triplets2.csv")
    
    #create_dataset(dataset_folder,output_file,filter_list = filter_list)
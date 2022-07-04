# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:54:55 2021

@author: prctha
"""
import pickle
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import OneHotEncoder
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import random
import os
import csv

# print(f"Torch version: {torch.__version__}")
# print(f"Cuda available: {torch.cuda.is_available()}")
# print(f"Torch geometric version: {torch_geometric.__version__}")


class CADDataset(Dataset):
    def __init__(self, root, filename, force_reprocess=False, transform=None, pre_transform=None, add_cats=False, triple_file=None):
        """
        root = Where the dataset should be stored. This folder is split into raw_dir (downloaded dataset) and processed_dir (processed dataset).
        Triple file needs to be csv, with three models per row, [base, match, not match]
        """
        self.filename = filename
        self.force_reprocess = force_reprocess
        self.add_cats = add_cats
        self.triple_file = triple_file
        self.filenamelist = []
        super(CADDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not trigged
        Download not implimented
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.force_reprocess == True:
            self.force_reprocess = False
            return 'reprocess.pt'
        
        ''' HR 01/06/22 Workaround to avoid FileNotFoundError '''
        print('self.processed_dir:', self.processed_dir)
        # folder,file = os.path.split(self.processed_dir)
        folder = self.processed_dir
        if not os.path.isdir(folder):
            print('  Making folder', folder)
            os.makedirs(folder)
        
        processedfiles = [f for f in os.listdir(self.processed_dir) if os.path.isfile(
            os.path.join(self.processed_dir, f))]
        if 'pre_filter.pt' in processedfiles:
            processedfiles.remove('pre_filter.pt')
        if 'pre_transform.pt' in processedfiles:
            processedfiles.remove('pre_transform.pt')
        # 'not_implimented.pt' #[f'data_{i}.pt' for i in list(self.data.index)]
        return processedfiles

    def download(self):
        pass

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data, _, _ = self[0]
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data, _, _ = self[0]
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")

    def write_triple_list(self, list_of_lists):
        # print(list_of_lists)
        with open(os.path.join(self.root, "triple_list.csv"), "w", newline='') as f:
            wr = csv.writer(f)
            wr.writerows(list_of_lists)
        self.filenamelist = list_of_lists

    def process(self):
        print(self.raw_paths)

        # ''' HR 01/06/22 Workaround to avoid FileNotFoundError '''
        # print('  self.raw_paths:', self.raw_paths)
        # folder,file = os.path.split(self.raw_paths[0])
        # print('  file, folder:', file, folder)
        # if not os.path.isdir(folder):
        #     print('  Making folder', folder)
        #     os.makedirs(folder)

        with open(self.raw_paths[0], 'rb') as f:
            dataset = pickle.load(f)

        # first create a dict of cats

        if self.triple_file != None:
            file = self.triple_file
            triple_list = []
            import csv
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row != []:
                        triple_list.append([row[0], row[1], row[2]])

            datatrip = {}
            for row in triple_list:
                try:
                    datatrip[row[0]] = dataset[row[0]]
                except:
                    print("Missing:",row[0])

        if self.add_cats:
            cat_dict = {}
            cat_vec = {}
            i = 0
            for name, graphdata in dataset.items():
                cat = graphdata['cat']
                if cat in cat_dict.keys():
                    cat_dict[cat].append(name)
                else:
                    cat_dict[cat] = [name]
                    cat_vec[i] = cat
                    i += 1

        # face type encoder, 10 types of face
        faceEnc = OneHotEncoder(categories=[range(0, 10)], sparse=False, drop='first')

        # edge type encoder, 10 types of face
        edgeEnc = OneHotEncoder(categories=[range(0, 10)], sparse=False, drop='first')

        data_list = []
        names_list = []
        i = -1
        # need to create groups of three, [root, same different]

        if self.triple_file == None:
            iterator = dataset.items()
        else:
            iterator = iter(triple_list)#datatrip.items()
        #print(len(iterator), len(triple_list))
        #assert len(iterator) == len(triple_list), "Hmmmm"
        
        test = False
        catstemp = ['Washers', 'Brackets', 'Gears', 'Nuts']

        for item1, item2 in tqdm(iterator): # 
            if self.triple_file != None:
                row = item1
                name = row[0]
            
            else:
                #print(item1)
                name = item1
                graphdata = item2
        #for row in tqdm(triple_list):
            #cat = dataset[name]['cat']
            #if cat not in catstemp:
            #    continue
            i += 1

            #print("name", name)

            if self.triple_file != None:
                if test == True:
                    assert name == triple_list[i][0], "Not a match"
                if name != triple_list[i][0]:
                    test = True
                    print("skipped", i)
                    continue

            x = None
            edge_index = None
            g = dataset[name]['graph_nx']
            cat = dataset[name]['cat']
            # g = graphdata['graph_nx']
            # cat = graphdata['cat']

            # find two other graphs in dataset
            # same type
            if self.add_cats:

                same_name = random.choice(cat_dict[cat])
                ran_cat = cat
                while ran_cat == cat:
                    ran_cat, ran_names = random.choice(list(cat_dict.items()))
                diff_name = random.choice(ran_names)

                name_list = [name, same_name, diff_name]
                names_list.append(name_list)

                g_s = dataset[same_name]['graph_nx']
                g_d = dataset[diff_name]['graph_nx']
            elif self.triple_file != None:
                same_name = row[1]
                diff_name = row[2]
                #print("match", same_name)
                #print("nonmatch", diff_name)
                g_s = dataset[same_name]['graph_nx']
                g_d = dataset[diff_name]['graph_nx']
                name_list = [name, same_name, diff_name]
                names_list.append(name_list)
            else:
                name_list = [name]
                names_list.append(name_list)

            if i == -100:
                visualize(g, 'red')
                visualize(g_s, 'green')
                visualize(g_d, 'blue')
            h = from_networkx(g)
            if self.add_cats or self.triple_file != None:
                h_s = from_networkx(g_s)
                h_d = from_networkx(g_d)

                H = [h, h_s, h_d]
            else:
                H = [h, h, h]
            # node_index = 0
            j = 0
            abc = ['a', 'b', 'c']
            for h_data in H:
                edge_index = h_data.edge_index

                # node features
                # 1. one hot encode face surface type
                x_enc = faceEnc.fit_transform(h_data.surfType.reshape(-1, 1))
                x = torch.from_numpy(x_enc)
                x = x.type(torch.FloatTensor)

                # edge_attr
                # 1. degrees between faces
                # 2. edge type
                # 3. relative face size
                # 4. edge length (not implimented need to recalculate this, based on proportion of total face edge length)

                # 1. degrees : values between (-180)-(180), scaled to between 0 and 1

                degrees = (h_data.degrees + 180)/360
                degrees = degrees.view(-1, 1)

                # 2. edgeType : 10 classes transfered into one hot vector

                edge_enc = edgeEnc.fit_transform(h_data.edgeType.reshape(-1, 1))
                edgeE = torch.from_numpy(edge_enc)
                edgeE = edgeE.type(torch.FloatTensor)

                # 4. relativeSize : releative size of faces between edges (needs scaling??)

                relativeSize = h_data.relativeSize
                relativeSize = relativeSize.view(-1, 1)

                # 4. length : (not implimented)

                # put features together

                edge_attr = torch.cat((degrees, edgeE, relativeSize), 1)

                # used to store similarity scores, but not needed if just 1 or 0
                # if j in [0,1]:
                #   y = torch.tensor([1])
                # else:
                #   y = torch.tensor([0])
                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                torch.save(graph_data, os.path.join(
                    self.processed_dir, 'data_{}.pt'.format(str(i) + abc[j])))
                # data_list.append(graph_data)
                j += 1
            # assert 1==0
        self.write_triple_list(names_list)

    def len(self):
        ll = len(self.processed_file_names)
        # if ll < 20:
            # print("self.processed_file_names",self.processed_file_names)
        # print("len(self.processed_file_names)",ll)
        # print("len(self.processed_file_names)/3",ll/3)
        return int(ll/3)

    def get(self, idx):
        abc = ['a', 'b', 'c']
        data0 = torch.load(os.path.join(self.processed_dir,
                                        'data_{}.pt'.format(str(idx) + abc[0])))
        data1 = torch.load(os.path.join(self.processed_dir,
                                        'data_{}.pt'.format(str(idx) + abc[1])))
        data2 = torch.load(os.path.join(self.processed_dir,
                                        'data_{}.pt'.format(str(idx) + abc[2])))
        return data0, data1, data2


if __name__ == "__main__":
    print("Starting dataset processing....")
    dataset = CADDataset(".\\Dataset", "ABC_nx.pickle", force_reprocess=True,
                         triple_file=".\\Dataset\\triple_data.csv")
    print("Dataset processed")

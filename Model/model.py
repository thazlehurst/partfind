# findfind model file
# TAH 2021

# SIMGNN

from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch
import dgl


class PartGNN(nn.Module):
    def __init__(self, args):
        super(PartGNN,self).__init__()
        
        self.args = args
        self.setup_layers()
        self.feature_count = self.args.tensor_neurons
        
        
    def setup_layers(self):
        """ Build layers of neural network"""
        
        self.conv1 = GraphConv(1,self.args.GNN_1)
        self.conv2 = GraphConv(self.args.GNN_1,self.args.GNN_2)
        self.conv3 = GraphConv(self.args.GNN_2,self.args.GNN_3)
        self.scoring_layer = nn.Linear(self.args.GNN_3,1)
        
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                        self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        
    
    def conv_pass(self,g):
        # """ g is single graph 
        # returns g
        # """
        # print("g",g)
        h = g.in_degrees().view(-1, 1).float()
        h = self.conv1(g,h)
        h = torch.nn.functional.relu(h)
        h = torch.nn.functional.dropout(h,p=self.args.dropout) # training=?
        h = self.conv2(g,h)
        h = torch.nn.functional.relu(h)
        h = torch.nn.functional.dropout(h,p=self.args.dropout) # training=?
        h = self.conv3(g,h) # with default values this is (n, 32)
        g.ndata['h'] = h
        # return g instead of h
        #print("h.shape",h.shape)
        return h # [batch_size, n, 32]
    
    def forward(self,G0,G1):
        """G0 is test graph
        G1 is matching
        G2 is not matching
        """
        
        features_G0 = self.conv_pass(G0)
        features_G1 = self.conv_pass(G1)
        #features_G2 = self.conv_pass(G2)
        
        # think th eproblem is here, maybe return g from conv_pass to properly process in attention
        
        pooled_features_G0 = self.attention(features_G0)
        pooled_features_G1 = self.attention(features_G1)
        #pooled_features_G2 = self.attention(features_G2)
        
        # print("pooled_features_G0",pooled_features_G0.shape)
        # print("pooled_features_G1",pooled_features_G1.shape)
        
        scores_1 = self.tensor_network(pooled_features_G0, pooled_features_G1)
        #scores_2 = self.tensor_network(pooled_features_G0, pooled_features_G2)
        scores_1 = torch.t(scores_1)
        #scores_2 = torch.t(scores_2)
        
        scores_1 = torch.nn.functional.relu(self.fully_connected_first(scores_1))
        #scores_2 = torch.nn.functional.relu(self.fully_connected_first(scores_2))
        
        score_1 = torch.sigmoid(self.scoring_layer(scores_1))
        #score_2 = torch.sigmoid(self.scoring_layer(scores_2))
        
        return score_1#, score_2
        
        # res = []
        
        # for i in range(2): #Siamese nets; sharing weights
            # # use node degree as inital features
            # h = g[i].in_degrees().view(-1, 1).float()
            # h = self.conv1(g[i], h)
            # h = torch.relu(h)
            # h = self.conv2(g[i], h)
            # g[i].ndata['h'] = h
            # # Calculate graph representation by averaging all the node representations.
            # hg = dgl.mean_nodes(g[i], 'h')
            # if i == 1:
                # output1 = F.relu(hg)
            # else:
                # output2 = F.relu(hg)
              # #res.append(F.relu(hg))
        # # res = torch.abs(res[1]-res[0])
        # # res = self.linear(res)
        # return output1, output2 # but maybe just h
        

class AttentionModule(nn.Module):
    def __init__(self,args):
        super(AttentionModule,self).__init__()
        self.args = args
        
        self.setup_weights()
        
        
    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.GNN_3,self.args.GNN_3))
        
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        
    # def forward(self, g_embedding):
        # batch_size = g_embedding.batch_size
        # unbatch_g = dgl.unbatch(g_embedding)
        # reps = []
        # for g in unbatch_g:
            # embedding = g.ndata['h']
            # global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
            # transformed_global = torch.tanh(global_context)
            # sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1,1)))
            # reps.append(torch.mm(torch.t(embedding), sigmoid_scores))
        # # representation = torch.cat(reps,dim=1)
        # representation = torch.stack(reps,dim=0)
        # return representation
        
    def forward(self, embedding):
        # embedding = g_embedding.ndata['h']
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1,1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation
        
        
class TenorNetworkModule(torch.nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.GNN_3,
                                                             self.args.GNN_3,
                                                             self.args.tensor_neurons))

        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.args.GNN_3))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.GNN_3, -1))
        scoring = scoring.view(self.args.GNN_3, self.args.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores
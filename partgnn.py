'''
PartGNN
T Hazlehurst
'''

'''
https://github.com/thazlehurst/partfind/blob/ab5f48d2697b1c219d1ecf4cff3efe9e1cdfa282/partgnn.py
'''


import torch
import time
import numpy as np
from pylab import zeros, arange, subplots, plt, savefig
from model import GCNTriplet
from CADDataset import CADDataset
from torch.nn import TripletMarginLoss, BCELoss
from torch_geometric.data import DataLoader


class PartGNN():

    def __init__(self, dataset, args):
        """
        :param args: Arguments object
        """
        super(PartGNN, self).__init__()
        self.args = args
        self.verbose = self.args.verbose
        
        self.save_frequency = 1
        self.save_folder = "./trained_models/"
        if self.verbose:
            print("Verbose enabled")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print("Using device:",self.device)
        
        self.training = True
        
        # load dataset
        self.dataset = dataset
        
        #load model
        try:
            self.convtype = self.args.convtype
        except:
            self.convtype = 'NNConv'
            print("Convolution type not defined using ",self.convtype)
        # convtype = "GraphConv"
        print("self.convtype:", self.convtype)
        self.model = GCNTriplet(hidden_channels=32,
                           dataset=dataset,
                           nb_layers=3,
                           convtype=self.convtype).to(self.device)
                           
        self.train_loader = None
        self.test_loader = None
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = TripletMarginLoss(margin=1.0, p=2)
        self.criterion2 = BCELoss()
        
    def load_data(self):
        if self.train_loader == None:
            dataset = self.dataset
            
            dataset = dataset.shuffle()

            split = 0.8
            train_split = int(len(dataset)*0.8)

            train_dataset = dataset[:train_split]
            test_dataset = dataset[train_split:]

            self.dataset_loaded = False


            print(f'Number of training graphs: {len(train_dataset)}')
            print(f'Number of test graphs: {len(test_dataset)}')

            batch_size = 32

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:
            pass
            
            
    def train(self):
        
        self.training = True
        torch.random.get_rng_state()
        
        epochs = 1+ self.args.epochs
        # Plotting config
        plot = True
        plot_data = {}
        plot_data['train_loss'] = zeros(epochs)
        plot_data['train_correct_triplets'] = zeros(epochs)
        plot_data['train_correct_percent'] = zeros(epochs)
        plot_data['train_correct_similarity'] = zeros(epochs)
        plot_data['train_correct_score'] = zeros(epochs)
        plot_data['val_loss'] = zeros(epochs)
        plot_data['val_correct_triplets'] = zeros(epochs)
        plot_data['val_correct_percent'] = zeros(epochs)
        plot_data['val_correct_similarity'] = zeros(epochs)
        plot_data['val_correct_score'] = zeros(epochs)
        plot_data['epoch'] = 0
        it_axes = arange(epochs)
        _, ax1 = subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('train loss (r), val loss (y)')
        ax2.set_ylabel('train correct pairs (b), val correct pairs (g)')
        ax2.set_autoscaley_on(False)
        ax1.set_ylim([0, 2])
        ax2.set_ylim([0, 1])
    
    
        for epoch in range(1, epochs):
            plot_data['epoch'] = epoch
            plot_data = self.__train(epoch, plot_data)
            plot_data = self.__validate(epoch, plot_data)

            if self.args.trainplot:
                ax1.plot(it_axes[0:epoch+1], plot_data['train_loss'][0:epoch+1], 'r')
                ax2.plot(it_axes[0:epoch+1],
                         plot_data['train_correct_percent'][0:epoch+1], 'b')

                ax1.plot(it_axes[0:epoch+1], plot_data['val_loss'][0:epoch+1], 'y')
                ax2.plot(it_axes[0:epoch+1],
                         plot_data['val_correct_percent'][0:epoch+1], 'g', '-')

                plt.title("Testing")
                plt.grid(True)
            
            self.save_model(epoch)
                
    def save_model(self,epoch):
        torch.save(self.model.state_dict(), str(self.args.trained_model) + self.convtype + "_epoch" + str(epoch).zfill(2) + "of" + str(self.args.epochs) + ".pt")

    def __train(self,epoch,plot_data):
    
        self.load_data()
        
        
        
        print(self.model)
        
        
        
        batch_time = self.AverageMeter()
        data_time = self.AverageMeter()
        loss_meter = self.AverageMeter()
        correct_triplets = self.AverageMeter()
        correct_percent = self.AverageMeter()
        correct_similarity = self.AverageMeter()
        correct_score = self.AverageMeter()

        self.model.train()
        
        
        end = time.time()
        # Iterate in batches over the training dataset.
        for i, (data0, data1, data2) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            data0 = data0.to(self.device)
            data1 = data1.to(self.device)
            data2 = data2.to(self.device)

            # data0 is anchor model, data1 is positive model, data2 is negative model
            if self.convtype in ['GCNConv', 'GraphConv']:  # node features only
              kwargs = {"x0": data0.x, "edge_index0": data0.edge_index, "batch0": data0.batch,
                        "x1": data1.x, "edge_index1": data1.edge_index, "batch1": data1.batch,
                        "x2": data2.x, "edge_index2": data2.edge_index, "batch2": data2.batch}
            elif self.convtype in ['NNConv']:  # node and edge features
              kwargs = {"x0": data0.x, "edge_index0": data0.edge_index, "batch0": data0.batch, "edge_attr0": data0.edge_attr,
                        "x1": data1.x, "edge_index1": data1.edge_index, "batch1": data1.batch, "edge_attr1": data1.edge_attr,
                        "x2": data2.x, "edge_index2": data2.edge_index, "batch2": data2.batch, "edge_attr2": data2.edge_attr}

            # Perform a single forward pass for each model.
            out0, out1, out2, correct, score_p, score_n, correct_s = self.model(**kwargs)

            loss = self.criterion(out0, out1, out2)  # Compute the losses.

            target_p = torch.ones(score_p.shape[0], 1).to(self.device)
            target_n = torch.zeros(score_p.shape[0], 1).to(self.device)

            # remove score element
            loss2a = self.criterion2(score_p, target_p)
            loss2b = self.criterion2(score_n, target_n)
            #print("loss",loss)
            a = torch.tensor(0.5, requires_grad=True)
            loss = (1-a)*loss + a*(loss2a+loss2b)

            # measure and record loss
            loss_meter.update(loss.data.item(), out1.size()[0])
            correct_triplets.update(torch.sum(correct))
            correct_percent.update(correct_triplets.val/out1.size()[0])
            correct_score.update(torch.sum(correct_s)/out1.size()[0])

            # calculate score errors
            pred = np.round(score_p.cpu().detach())
            target = np.round(target_p.cpu().detach())
            acc_p = self.__accuracy_score(target, pred)
            pred = np.round(score_p.cpu().detach())
            target = np.round(target_p.cpu().detach())
            acc_n = self.__accuracy_score(target, pred)
            #print("pos acc",acc_p,"neg acc",acc_n)

            acc = (acc_p + acc_n)/2
            correct_similarity.update(acc)

            #print("loss",loss)
            loss.backward()  # Derive gradients.
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Correct Triplets {correct_triplets.val:.3f} ({correct_triplets.avg:.3f})\t'
                      'Correct Percent {correct_percent.val:.3f} ({correct_percent.avg:.3f}) \t'
                      'Sim acc {correct_similarity.val:.3f} ({correct_similarity.avg:.3f}) \t'
                      'score acc {correct_score.val:.3f} ({correct_score.avg:.3f}) \t'
                      .format(
                          epoch, i, len(self.train_loader), batch_time=batch_time,
                          data_time=data_time, loss=loss_meter, correct_triplets=correct_triplets, correct_percent=correct_percent,
                          correct_similarity=correct_similarity, correct_score=correct_score))

                # break
                #assert 1==0

        plot_data['train_loss'][plot_data['epoch']] = loss_meter.avg
        plot_data['train_correct_triplets'][plot_data['epoch']] = correct_triplets.avg
        plot_data['train_correct_percent'][plot_data['epoch']] = correct_percent.avg
        plot_data['train_correct_similarity'][plot_data['epoch']
                                              ] = correct_similarity.avg
        plot_data['train_correct_score'][plot_data['epoch']] = correct_score.avg
        return plot_data

    def __accuracy_score(self,true, pred):
        correct = 0
        for i, _ in enumerate(pred):
            if true[i] == pred[i]:
                correct += 1
        return correct/len(pred)
        
            
    def __validate(self,epoch, plot_data, print_freq=1):
        batch_time = self.AverageMeter()
        data_time = self.AverageMeter()
        loss_meter = self.AverageMeter()
        correct_triplets = self.AverageMeter()
        correct_percent = self.AverageMeter()
        correct_similarity = self.AverageMeter()
        correct_score = self.AverageMeter()
        
        self.load_data()
        
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            # Iterate in batches over the training dataset.
            for i, (data0, data1, data2) in enumerate(self.test_loader):

                # measure data loading time
                data_time.update(time.time() - end)

                data0 = data0.to(self.device)
                data1 = data1.to(self.device)
                data2 = data2.to(self.device)

                # data0 is anchor model, data1 is positive model, data2 is negative model
                if self.convtype in ['GCNConv', 'GraphConv']:  # node features only
                    kwargs = {"x0": data0.x, "edge_index0": data0.edge_index, "batch0": data0.batch,
                              "x1": data1.x, "edge_index1": data1.edge_index, "batch1": data1.batch,
                              "x2": data2.x, "edge_index2": data2.edge_index, "batch2": data2.batch}
                elif self.convtype in ['NNConv']:  # node and edge features
                    kwargs = {"x0": data0.x, "edge_index0": data0.edge_index, "batch0": data0.batch, "edge_attr0": data0.edge_attr,
                              "x1": data1.x, "edge_index1": data1.edge_index, "batch1": data1.batch, "edge_attr1": data1.edge_attr,
                              "x2": data2.x, "edge_index2": data2.edge_index, "batch2": data2.batch, "edge_attr2": data2.edge_attr}

                # Perform a single forward pass for each model.
                out0, out1, out2, correct, score_p, score_n, correct_s = self.model(**kwargs)

                # out0 = model(data0.x, data0.edge_index, data0.batch)  # Perform a single forward pass for each model.
                # out1 = model(data1.x, data1.edge_index, data1.batch)  # Perform a single forward pass for each model.
                # out2 = model(data2.x, data2.edge_index, data2.batch)  # Perform a single forward pass for each model.

                loss = self.criterion(out0, out1, out2)  # Compute the loss.
                #print("loss",loss)

                target_p = torch.ones(score_p.shape[0], 1).to(self.device)
                target_n = torch.zeros(score_p.shape[0], 1).to(self.device)

                loss2a = self.criterion2(score_p, target_p)
                loss2b = self.criterion2(score_n, target_n)
                # print("loss",loss)
                a = torch.tensor(0.5, requires_grad=True)
                loss = (1-a)*loss + a*(loss2a+loss2b)

                # measure and record loss
                loss_meter.update(loss.data.item(), out1.size()[0])
                correct_triplets.update(torch.sum(correct))
                #correct_percent = 100*correct_triplets.val/out1.size()[0]
                correct_percent.update(correct_triplets.val/out1.size()[0])
                correct_score.update(torch.sum(correct_s)/out1.size()[0])

                # calculate score errors
                pred = np.round(score_p.cpu().detach())
                target = np.round(target_p.cpu().detach())
                acc_p = self.__accuracy_score(target, pred)
                pred = np.round(score_n.cpu().detach())
                target = np.round(target_n.cpu().detach())
                acc_n = self.__accuracy_score(target, pred)
                print("pos acc", acc_p, "neg acc", acc_n)

                acc = (acc_p + acc_n)/2
                correct_similarity.update(acc)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print('Validation: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Correct Triplets {correct_triplets.val:.3f} ({correct_triplets.avg:.3f})\t'
                          'Correct Percent {correct_percent.val:.3f} ({correct_percent.avg:.3f}) \t'
                          'Sim acc {correct_similarity.val:.3f} ({correct_similarity.avg:.3f}) \t'
                          'score acc {correct_score.val:.3f} ({correct_score.avg:.3f}) \t'
                          .format(
                              epoch, i, len(self.test_loader), batch_time=batch_time,
                              data_time=data_time, loss=loss_meter, correct_triplets=correct_triplets, correct_percent=correct_percent,
                              correct_similarity=correct_similarity, correct_score=correct_score))

                  # break

        plot_data['val_loss'][plot_data['epoch']] = loss_meter.avg
        plot_data['val_correct_triplets'][plot_data['epoch']] = correct_triplets.avg
        plot_data['val_correct_percent'][plot_data['epoch']] = correct_percent.avg
        plot_data['val_correct_similarity'][plot_data['epoch']
                                              ] = correct_similarity.avg
        plot_data['val_correct_score'][plot_data['epoch']] = correct_score.avg

        return plot_data
        

    def load_model(self,model_loc=None):
        if model_loc == None:
            model_path = self.args.model_loc
        else:
            model_path = model_loc
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()
        print("Model loaded")

    def get_vectors(self,models_dataset):
        self.training = False
        torch.manual_seed(42)
        print("Getting vectors...")
        
        self.model.eval()
        
        array = np.empty((0,16), int)
        with torch.no_grad():
            # Iterate in batches over the training dataset.
            for i, (data0, _, _) in enumerate(models_dataset):

                data0 = data0.to(self.device)
                data1 = data0.to(self.device)
                data2 = data0.to(self.device)

                # data0 is anchor model, data1 is positive model, data2 is negative model
                if self.convtype in ['GCNConv', 'GraphConv']:  # node features only
                    kwargs = {"x0": data0.x, "edge_index0": data0.edge_index, "batch0": data0.batch,
                              "x1": data1.x, "edge_index1": data1.edge_index, "batch1": data1.batch,
                              "x2": data2.x, "edge_index2": data2.edge_index, "batch2": data2.batch}
                elif self.convtype in ['NNConv']:  # node and edge features
                    kwargs = {"x0": data0.x, "edge_index0": data0.edge_index, "batch0": data0.batch, "edge_attr0": data0.edge_attr,
                              "x1": data1.x, "edge_index1": data1.edge_index, "batch1": data1.batch, "edge_attr1": data1.edge_attr,
                              "x2": data2.x, "edge_index2": data2.edge_index, "batch2": data2.batch, "edge_attr2": data2.edge_attr}

                # Perform a single forward pass for each model.
                out0, out1, out2, correct, score_p, score_n, correct_s = self.model(**kwargs)
                
                array = np.append(array, np.array(out0), axis=0)

        return array


    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

### use soft max for similarity score
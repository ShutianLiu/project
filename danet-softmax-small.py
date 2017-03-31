
# coding: utf-8

# In[1]:

from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import numpy as np
import h5py
import time
import subprocess

import torch_utils
import data_utils

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

#get_ipython().magic('env CUDA_VISIBLE_DEVICES=0')


# In[2]:

# prepare exp folder
#base_dir='/hdd2/yluo/pytorch/'  # change this!
#exp_name='DANet-softmax'  # change this!
base_dir='/home/shutian/Desktop/Project/'  # change this!
exp_name='DANet-softmax-small'  # change this!
exp_prepare='/hdd2/yluo/exp_prepare_folder.sh'

net_dir=base_dir+'/exp/'+exp_name+'/net'

#subprocess.call(exp_prepare+ ' ' + exp_name+ ' ' +base_dir, shell=True)

# global params

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--batch-size', type=int, default=4,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training (default: True)')
parser.add_argument('--seed', type=int, default=20170220,
                    help='random seed (default: 20170220)')
parser.add_argument('--infeat-dim', type=int, default=300,
                    help='dimension of the input feature (default: 129)')
parser.add_argument('--outfeat-dim', type=int, default=20,
                    help='dimension of the embedding (default: 20)')
parser.add_argument('--seq-len', type=int, default=100,
                    help='length of the sequence (default: 100)')
parser.add_argument('--log-step', type=int, default=100,
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--num-layers', type=int, default=4,
                    help='number of stacked RNN layers (default: 1)')
parser.add_argument('--bidirectional', action='store_true', default=True,
                    help='whether to use bidirectional RNN layers (default: True)')
parser.add_argument('--val-save', type=str,  default=base_dir+'exp/'+exp_name+'/net/cv/model.pt',
                    help='path to save the best model')

args, _ = parser.parse_known_args()
args.cuda = args.cuda and torch.cuda.is_available()
args.num_direction = int(args.bidirectional)+1

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} 
else:
    kwargs = {}
    
#training_data_path = '/hdd2/yluo/dpcl_tf/dataset/e2e/hard/new/wsj_tr_100'  # change this!
#validation_data_path = '/hdd2/yluo/dpcl_tf/dataset/e2e/hard/new/wsj_cv_100'  # change this!
training_data_path = '/home/shutian/Desktop/Project/data/small/data_tr_300'  # change this!
validation_data_path = '/home/shutian/Desktop/Project/data/small/data_cv_300'  # change this!



# In[3]:

# define data loaders

train_loader = DataLoader(data_utils.WSJDataset(training_data_path), 
                          batch_size=args.batch_size, 
                          shuffle=True, 
                          **kwargs)

validation_loader = DataLoader(data_utils.WSJDataset(validation_data_path), 
                          batch_size=args.batch_size, 
                          shuffle=False, 
                          **kwargs)

args.dataset_len = len(train_loader)
args.log_step = 1#args.dataset_len // 1 #4


# In[4]:

# define model

class DANet(nn.Module):
    def __init__(self):
        super(DANet, self).__init__()
        
        self.rnn = torch_utils.MultiRNN('LSTM', args.infeat_dim, 300, 
                                           num_layers=args.num_layers, 
                                           bidirectional=args.bidirectional)
        self.FC = torch_utils.FCLayer(600, args.infeat_dim*args.outfeat_dim, nonlinearity='tanh')
        
    def forward(self, input, ibm, hidden):  #input=batch_infeat   ibm=batch_ibm
        seq_len = input.size(1)
        #print('seq_len:')
	#print(seq_len)

        output, hidden = self.rnn(input, hidden)
	#print('1output.size:')
	#print(output.size())
        output = output.contiguous()  # batch, T, 4*H
	#print('2output.size:')
	#print(output.size())
        output = output.view(-1, output.size(2))  # batch*T, 4*H        -1 means dimension can be inferred
	#print('3output.size:')
	#print(output.size())
        output = self.FC(output)  # batch*T, F*N    F=H   N=20
	#print('4output.size:')
	#print(output.size())

        V1 = output.view(-1, seq_len*args.infeat_dim, args.outfeat_dim)  # batch, T*F, N
        
	
	
        VY_batch = torch.bmm(torch.transpose(V1, 1,2), ibm)  # batch, N, nspk
	#print('VY_batch.size:')
	#print(VY_batch.size())
        sum_batch = torch.sum(ibm, 1).view(-1, 1, ibm.size(2))  # batch, 1, nspk
        sum_batch = sum_batch.expand(sum_batch.size(0), V1.size(1), sum_batch.size(2))
        affinity_mask = torch.bmm(V1, VY_batch) / (sum_batch + 1e-8)  # batch, T*F, nspk
        #print('1affinity_mask.size:')
	#print(affinity_mask.size())
        affinity_mask = affinity_mask.view(-1, affinity_mask.size(2))
        affinity_mask = F.softmax(affinity_mask)
        affinity_mask = affinity_mask.view(-1, seq_len*args.infeat_dim, affinity_mask.size(1))  # batch, T*F, nspk
	#print('4affinity_mask.size:')
	#print(affinity_mask.size())        
	
        return affinity_mask, hidden
    
    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)
        
    
def objective(mixture, wf, affinity_mask):
    #print('mixture.size()')    
    #print(mixture.size())
    loss = mixture.expand(mixture.size(0), mixture.size(1), wf.size(2)) * (wf - affinity_mask)
    #print('In function objective()')
    #print(mixture)
    #print(wf[1,100,1])
    #print(affinity_mask[1,100,1])
    #print(loss[1,1000:2000])
    #print('loss.size()')
    #print(loss.size())
    #print(loss[1,3000:4000])
    loss = loss.view(-1, loss.size(1)*loss.size(2))
    #print('loss.size():')
    #print(loss.size())
    #print(loss[1,3000:4000])
    return torch.mean(torch.sum(torch.pow(loss, 2), 1))
    
model = DANet()
if args.cuda:
    model.cuda()
    
optimizer = optim.Adam(model.parameters(), lr=args.lr)
lr_decaytime = 0

# In[5]:

# function for training and validation

def train(epoch):
    start_time = time.time()
    ########
    #print('aaaaaaaaaaaaaaaaaa')
    model.train()  
    #print('aaaaaaaaaaaaaaaaaa')
    train_loss = 0.
    
    for batch_idx, data in enumerate(train_loader): #data=train_loader[batch_idx]
	
        batch_infeat = Variable(data[0]).contiguous()  # batch, T, F
	#print('batch_infeat size:')
	#print(batch_infeat.size())
	#print(batch_infeat.data)
	
	#print(np.argwhere(np.isnan(batch_infeat.data)))

        batch_wf = Variable(data[1]).contiguous()
	#print('wf 1')
	#print(batch_wf.size())
	#print(np.argwhere(np.isnan(batch_wf)))
	#print(batch_wf.data)
        batch_wf = batch_wf.view(batch_wf.size(0), batch_wf.size(3), batch_wf.size(2)*batch_wf.size(1))
	#print('wf 2')
	#print(batch_wf.size())
        batch_wf = torch.transpose(batch_wf, 1,2)  # batch, TF, 4
	#print('wf 3')
	#print(batch_wf.size())
	#print(batch_wf)
        batch_mix = Variable(data[2]).contiguous()
	#print('batch_mix size:')
	#print(batch_mix.size())
        #print(np.argwhere(np.isnan(batch_mix)))
	#print(batch_mix.data)
        batch_mix = batch_mix.view(batch_mix.size(0), -1, 1)  # batch, TF, 1
	
        batch_ibm = Variable(data[3]).contiguous()
	#print('ibm.shape')
	#print(batch_ibm.size())
    	#print(np.argwhere(np.isnan(batch_ibm)))
	#print(batch_ibm.data)
        batch_ibm = batch_ibm.view(batch_ibm.size(0), batch_ibm.size(3), batch_ibm.size(2)*batch_ibm.size(1))
        batch_ibm = torch.transpose(batch_ibm, 1,2)  # batch, TF, 4
	#print('batch_ibm size:')
	#print(batch_ibm)
        if args.cuda:
            batch_infeat = batch_infeat.cuda()
            batch_wf = batch_wf.cuda()
            batch_mix = batch_mix.cuda()
            batch_ibm = batch_ibm.cuda()
        
        hidden = model.init_hidden(batch_infeat.size(0))
        optimizer.zero_grad()
        #print('bbbbbbbbbb')
	#print(batch_infeat.data)
	#print(batch_ibm.data)
        output_mask, hidden = model(batch_infeat, batch_ibm, hidden)
	#print('output_mask')
	#print(output_mask)
	#print('hidden')
	#print(hidden)
        #print('bbbbbbbbbb')
	#print(batch_mix.data)
	#print(batch_wf.data)
	#print(output_mask.data)
        loss = objective(batch_mix, batch_wf, output_mask)
        #print('cccccccccc')
	#print('loss calculated :')
	#print(loss)
        loss.backward()
        #print('dddddddddd')
        train_loss += loss.data[0]
        #print('eeeeeeeeee')
        optimizer.step()
        #print('fffffffffff')
        if (batch_idx+1) % args.log_step == 0:
            ###############
            #print('batch_idx+1 :')
            #print(batch_idx+1)
            #############
            elapsed = time.time() - start_time
	    
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx+1, len(train_loader),
                elapsed * 1000 / (batch_idx+1), train_loss /( batch_idx+1)))
    	if epoch %10 ==0:
		print(output_mask)	
	    #print('loss :')
	    #print(train_loss)
    #print('ggggggggggggg')
    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), train_loss))
    
    return train_loss
        
def test(epoch):
    start_time = time.time()
    model.eval()
    validation_loss = 0.
    #print('in function test():')
    for batch_idx, data in enumerate(validation_loader):
        batch_infeat = Variable(data[0]).contiguous()
	
        batch_wf = Variable(data[1]).contiguous()
        batch_wf = batch_wf.view(batch_wf.size(0), batch_wf.size(3), batch_wf.size(2)*batch_wf.size(1))
        batch_wf = torch.transpose(batch_wf, 1,2)
        batch_mix = Variable(data[2]).contiguous()
        batch_mix = batch_mix.view(batch_mix.size(0), -1, 1)
        batch_ibm = Variable(data[3]).contiguous()
        batch_ibm = batch_ibm.view(batch_ibm.size(0), batch_ibm.size(3), batch_ibm.size(2)*batch_ibm.size(1))
        batch_ibm = torch.transpose(batch_ibm, 1,2)
        if args.cuda:
            batch_infeat = batch_infeat.cuda()
            batch_wf = batch_wf.cuda()
            batch_mix = batch_mix.cuda()
            batch_ibm = batch_ibm.cuda()
        
        hidden = model.init_hidden(batch_infeat.size(0))
        output_mask, hidden = model(batch_infeat, batch_ibm, hidden)
        #print('size of infeat, ibm, hidden')
	#print(batch_infeat.size(),batch_ibm.size())
	#print('size of mix wf ')
	#print(batch_mix.size(),batch_wf.size())
        loss = objective(batch_mix, batch_wf, output_mask)
        validation_loss += loss.data[0]
    	
    validation_loss /= (batch_idx+1)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)
    
    return validation_loss


# In[6]:

# main
training_loss = []
validation_loss = []
for epoch in range(1, args.epochs + 1):
    training_loss.append(train(epoch))
    validation_loss.append(test(epoch))
    if training_loss[-1] == np.min(training_loss):
        print('      Best training model found.')
        print('-' * 99)
    if validation_loss[-1] == np.min(validation_loss):
        # save current best model
        with open(args.val_save, 'wb') as f:
            torch.save(model, f)
            print('      Best validation model found and saved.')
            print('-' * 99)
    # lr decay
    if np.min(training_loss) not in training_loss[-3:]:
        #current_lr *= 0.5
	lr_decaytime +=1
        current_lr = args.lr*(0.5)**lr_decaytime
        optimizer = optim.RMSprop(model.parameters(), lr=current_lr)
        print('      Learning rate decreased.')
        print('-' * 99)
        


# In[ ]:




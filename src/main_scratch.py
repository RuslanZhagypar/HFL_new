#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import random

# import matplotlib

# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import time
import datetime
import torchvision.models as models
# from resnet import resnet20

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNEMnist, CNNCifar, ResNet
from models.Fed import FedAvg, WAvg
from models.test import test_img

from data_reader import femnist, celeba

import wandb
wandb.init(project="NEW_PROJECT_HFL")

# Set random seed for Python's built-in random module
random.seed(55)

# Set random seed for NumPy
np.random.seed(55)

# Set random seed for PyTorch
torch.manual_seed(55)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



Power = [0.2963  ,  0.2943  ,  0.2960   , 0.2980   , 0.2939  ,  0.2946 ,   0.2953    ,0.2965  ,  0.2951  ,  0.2946] # time charge = 40 min




           
U = [0.653442, 0.22041, 0.564881, 0.989151, 0.187596, 0.256222, 0.385553, 0.715504, 0.344045, 0.261993]


# joint uav 1 <-> devices
U0_devs = [8,10,18,26,44]
Joint_U0 = [0.945309,0.975503,0.531302,0.956634,0.990661]

# joint uav 2 <-> devices
U1_devs = [3,21,27,30,39,40]
Joint_U1 = [0.988247,0.617819,0.890511,0.961379,0.431,0.990943]

# joint uav 3 <-> devices
U2_devs = [7,12,15,19,25,38]
Joint_U2 = [0.658034,0.973349,0.987113,0.587312,0.95735,0.865958]

# joint uav 4 <-> devices
U3_devs = [1,6,33,42]
Joint_U3 = [0.988997,0.60033,0.943145,0.813693]

# joint uav 5 <-> devices
U4_devs = [2,14,16,17,32,46]
Joint_U4 = [0.889481,0.816955,0.990322,0.984967,0.975783,0.902254]

# joint uav 6 <-> devices
U5_devs = [13,24,35,47]
Joint_U5 = [0.692213,0.98646,0.969019,0.986819]

# joint uav 7 <-> devices
U6_devs = [0,9,22,29,31,37,43,45,49]
Joint_U6 =  [0.271776,0.929762,0.957161,0.614179,0.408366,0.981904,0.884558,0.845861,0.973165]

# joint uav 8 <-> devices
U7_devs = [4,36,41,48]
Joint_U7 = [0.959092,0.990111,0.903782,0.954914]

# joint uav 9 <-> devices
U8_devs = [5,11,28,34]
Joint_U8 = [0.938734,0.988103,0.937109,0.971837]

# joint uav 10 <-> devices
U9_devs = [20,23]
Joint_U9 = [0.85903,0.989588]


UAV_devs_dict = {
    "U0_devs": U0_devs,
    "U1_devs": U1_devs,
    "U2_devs": U2_devs,
    "U3_devs": U3_devs,
    "U4_devs": U4_devs,
    "U5_devs": U5_devs,
    "U6_devs": U6_devs,
    "U7_devs": U7_devs,
    "U8_devs": U8_devs,
    "U9_devs": U9_devs}

Joint_U_dict = {
    "Joint_U0": Joint_U0,
    "Joint_U1": Joint_U1,
    "Joint_U2": Joint_U2,
    "Joint_U3": Joint_U3,
    "Joint_U4": Joint_U4,
    "Joint_U5": Joint_U5,
    "Joint_U6": Joint_U6,
    "Joint_U7": Joint_U7,
    "Joint_U8": Joint_U8,
    "Joint_U9": Joint_U9}

# for i, j in enumerate(UAV_devs_dict):
#     JointU = Joint_U_dict['Joint_U' + str(UAV_index)]


if __name__ == '__main__':
    # parse args
    for iterAVG in range(10):

        args = args_parser()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.device = 'cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu'
        # args.device = 'cpu'
        print("DEVICE: ",args.device )

        # load dataset and split users
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        else:
            exit('Error: unrecognized dataset')
        img_size = dataset_train[0][0].shape

        # build model
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
            net_glob = nn.DataParallel(net_glob)
        elif args.model == 'vgg':
            net_glob = models.vgg11(pretrained=False).to(args.device)
            net_glob = nn.DataParallel(net_glob)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'femnist':
            net_glob = CNNEMnist(args=args).to(args.device)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=300, dim_out=args.num_classes)
            net_glob = nn.DataParallel(net_glob)
            net_glob = net_glob.to(args.device)
        else:
            exit('Error: unrecognized model')
        print(net_glob)
        net_glob.train()  # net initialization



        # initialization
        loss_train = []
        acc_train = []
        test_acc_train = []
        test_loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []


        #----------------------------------------

        lweight = []
        for k in range(args.num_users):
            lweight.append(len(dict_users[k])/60000)
            # SUM+= len(dict_users[k])

        # ----------------------------------------
        gweight = []
        for u in range(args.num_groups):
            p_per_uav = 0
            for k in UAV_devs_dict['U' + str(u) + '_devs']:
                p_per_uav = p_per_uav + lweight[k]
            gweight.append(p_per_uav)

        # print(sum(gweight))
        # group params
        num_samples = np.array([len(dict_users[i]) for i in dict_users])
        num_clients = int(args.num_users / args.num_groups)
        num_sampled = int(np.round(args.num_users * args.frac))
        group = []
        # gweight = []
        num_sampled_clients = np.ones(args.num_groups, dtype=int) * int(num_clients * args.frac)
        num_sampled_clients[args.num_groups - 1] = num_sampled - np.sum(num_sampled_clients[0:args.num_groups - 1])
        # lweight = []
        for i in range(0, args.num_groups):
            if i == args.num_groups - 1:
                group.append(np.arange(i * num_clients, args.num_users))
                # gweight.append(np.sum(num_samples[i * num_clients:args.num_users]) / np.sum(num_samples))
                # lweight.append(num_samples[i*num_clients:args.num_users]/np.sum(num_samples[i*num_clients:(i+1)*num_clients]))
            else:
                group.append(np.arange(i * num_clients, (i + 1) * num_clients))
                # gweight.append(np.sum(num_samples[i * num_clients:(i + 1) * num_clients]) / np.sum(num_samples))
                # lweight.append(num_samples[i*num_clients:(i+1)*num_clients]/np.sum(num_samples[i*num_clients:(i+1)*num_clients]))
        
        # different I for different groups can be set here.
        group_epochs = np.ones(args.num_groups, dtype=int) * args.group_freq
        local_iters = np.ones(args.num_groups, dtype=int) * args.local_period
        # the number of local iterations can not exceed the number of local batches.

        # training

        # filename
        f_l1 = args.dataset + '-' + str(args.num_groups) + '-' + str(args.group_freq) + '-' + str(
            args.local_period) + '-loss.txt'
        f_a1 = args.dataset + '-' + str(args.num_groups) + '-' + str(args.group_freq) + '-' + str(
            args.local_period) + '-acc.txt'
        f_l2 = args.dataset + '-' + str(args.num_groups) + '-' + str(args.group_freq) + '-' + str(
            args.local_period) + '-test_loss.txt'
        f_a2 = args.dataset + '-' + str(args.num_groups) + '-' + str(args.group_freq) + '-' + str(
            args.local_period) + '-test_acc.txt'

        stime = time.time()

        for iters in range(args.epochs):
            # copy weights
            w_glob = net_glob.state_dict()
            net_glob_v2 = copy.deepcopy(net_glob).to(args.device)
            w_glob_temp = net_glob_v2.state_dict()
            for k in w_glob_temp.keys():
                w_glob_temp[k] = 0

            S = np.random.choice(args.num_groups, size=(1, args.num_groups), replace=False)  # Sampling (Scheduling)
            temp = np.random.rand(1, args.num_groups)  # temp.shape = (1,M)
            activeUAVs = [s for k, s in enumerate(S[0]) if temp[0, s] < Power[s]]  # returns a list of the participants' indices
            
#             with open('available_UAVs_0_r1.txt', 'a') as file:
#                 # Convert each item in the list to a string and join them with commas
#                 line = ', '.join(map(str, activeUAVs))
#                 # Write the joined string followed by a newline character
#                 file.write(line + '\n')
                
                
            S = np.random.choice(args.num_groups, size=(1, args.num_groups), replace=False)  # Sampling (Scheduling)
            temp = np.random.rand(1, args.num_groups)  # temp.shape = (1,M)
            activeUAVs = [s for k, s in enumerate(activeUAVs) if temp[0, s] < U[s]]  # returns a list of the participants' indices

            # with open('Devs_after_power_50Wh_V2.txt', 'a') as file:
            #     # Convert each item in the list to a string and join them with commas
            #     line = ', '.join(map(str, activeUAVs))
            #     # Write the joined string followed by a newline character
            #     file.write(line + '\n')
            np.random.seed(60+iters+iterAVG*20)

            # for group_idx in range(args.num_groups):
            for group_idx in activeUAVs:
                # print("UAV ", group_idx)
                # print('***** group_idx', group_idx)
                net_group = copy.deepcopy(net_glob).to(args.device)
                net_group_v2 = copy.deepcopy(net_glob).to(args.device)

                loss_locals, acc_locals = [], []
                # num_sampled = int(len(group[group_idx])*args.frac)
                w_group = net_group.state_dict()
                w_group_local = net_group_v2.state_dict()
                for i in range(0, group_epochs[group_idx]):

                    UAV_devs = UAV_devs_dict['U' + str(group_idx) + '_devs']  # list of devices in the chosen UAV cell
                    JointU = Joint_U_dict['Joint_U' + str(group_idx)]

                    # S_dev = np.random.choice(UAV_devs, size=(1, len(UAV_devs)), replace=False)  # Sampling (Scheduling)
                    S_dev = UAV_devs  # Sampling (Scheduling)
                    temp = np.random.rand(1, len(UAV_devs))  # temp.shape = (1,M)
                    activeDevs_per_UAV = [s for k, s in enumerate(S_dev) if temp[0, k] < JointU[k]]  # returns a list of the participants' indices
                    
                    
                    net_group_local_v2 = copy.deepcopy(net_group).to(args.device)
                    w_local_previous = net_group_local_v2.state_dict()

                    net_group_local_v3 = copy.deepcopy(net_group).to(args.device)
                    w_local_temp = net_group_local_v3.state_dict()
                    for k in w_local_temp.keys():
                        w_local_temp[k] = 0

                    for j in activeDevs_per_UAV:
                        idx = j
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                                            iters=local_iters[group_idx], nums=num_samples[idx])
                        net_group_local = copy.deepcopy(net_group).to(args.device)
                        w_temp_print = net_group_local.state_dict() #delete this line

                        w = local.train(net=net_group_local)

                        for k in w.keys():
                            w[k] = w[k].detach()

                        if w_group is None:
                            # w_group = w
                            # for k in w.keys():
                            #     w_group[k] *= lweight[j]
                            pass
                        else:
                            index_0 = UAV_devs.index(j)
                            index = UAV_devs.index(j)
                            # print("index: ",index)
                            for k in w.keys():
                                w_local_temp[k] += (lweight[idx]/gweight[group_idx]) * (w[k] - w_local_previous[k])
                                # w_local_temp[k] += (lweight[idx]/gweight[group_idx])/(JointU[index_0]) * (w[k] - w_local_previous[k])
                                # w_group[k] += (lweight[index]/gweight[group_idx])/(JointU[index_0]) * (w[k] - w_local_previous[k])
                                # w_group[k] += (lweight[index]/gweight[group_idx])/(JointU[index_0]) * (w[k] - w_group[k])
                                # w_group[k] += (lwe§ight[index]/gweight[group_idx]) * (w[k] - w_group[k])
                        del net_group_local
                    for k in w_glob.keys():
                        w_group[k] += w_local_temp[k]
                    net_group.load_state_dict(w_group)

                # w_groups.append(w_group)
                if w_glob is None:
                    # w_glob = w_group
                    # for k in w_group.keys():
                    #     w_glob[k] *= gweight[group_idx]
                    pass
                else:
                    # w_glob =
                    for k in w_group.keys():
                        w_glob_temp[k] += gweight[group_idx] * (w_group[k] - w_group_local[k])
                        # w_glob_temp[k] += gweight[group_idx]/(U[group_idx]) * (w_group[k] - w_group_local[k])
                        # w_glob_temp[k] += gweight[group_idx]/(Power[group_idx]*U[group_idx]) * (w_group[k] - w_group_local[k])
                        # w_glob[k] += gweight[group_idx] * (w_group[k] - w_glob[k])
            for k in w_glob.keys():
                w_glob[k] += w_glob_temp[k]
                # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # compute training/test accuracy/loss
            if (iters + 1) % args.interval == 0:
                with torch.no_grad():
                    acc_avg, loss_avg = test_img(copy.deepcopy(net_glob).to(args.device), dataset_train, args)
                    acc_test, loss_test = test_img(copy.deepcopy(net_glob).to(args.device), dataset_test, args)
                print('Round {:3d}, Training loss {:.3f}'.format(iters, loss_avg), flush=True)
                print('Round {:3d}, Training acc {:.3f}'.format(iters, acc_avg), flush=True)
                print('Round {:3d}, Test loss {:.3f}'.format(iters, loss_test), flush=True)
                print('Round {:3d}, Test acc {:.3f}'.format(iters, acc_test), flush=True)

                wandb.log({'Accuracy': acc_test/100})
                wandb.log({'Loss': loss_test})
                # wandb.log({'Loss': loss_avg})

                # loss_train.append(loss_avg)
                # acc_train.append(acc_avg)
                # test_loss_train.append(loss_test)
                # test_acc_train.append(acc_test)

                # write into files
                # with open(f_l1, 'a') as l1, open(f_a1, 'a') as a1, open(f_l2, 'a') as l2, open(f_a2, 'a') as a2:
                #     l1.write(str(loss_avg))
                #     l1.write('\n')
                #     a1.write(str(acc_avg))
                #     a1.write('\n')
                #     l2.write(str(loss_test))
                #     l2.write('\n')
                #     a2.write(str(acc_test))
                #     a2.write('\n')

        ftime = time.time() - stime
        ftime = datetime.timedelta(seconds=ftime)
        print("Training time {}".format(ftime))
        # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-loss.txt',loss_train)
        # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-acc.txt',acc_train)
        # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-test_loss',test_loss_train)
        # np.savetxt(str(args.num_groups)+'-'+str(args.group_freq)+'-'+str(args.local_period)+'-test_acc',test_acc_train)






from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import PIL
import time
import math 
import shutil
import random
import warnings
import argparse

import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from torchsummary import summary
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.models as models
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import csv
import json
import pickle
import itertools
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
#from pytorch_lightning.metrics.classification import PrecisionRecall

from config import _C as cnfg
from utils_prog import progress_bar
from hr_capsule_model import HRCapsModel 

import xlwt 
from xlwt import Workbook 
np.seterr(divide='ignore', invalid='ignore')

#from augmentation_utils import train_transform, val_transform
#from strong_transform import strong_aug

#Network Parameters
def configure():
    cfg = 'path to "config.yaml"'
    cnfg.defrost()
    cnfg.merge_from_file(cfg)
    cnfg.freeze()

# create model
def model_build():
    #cudnn stuff
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = cnfg.cudnn.deterministic
    torch.backends.cudnn.enabled = cnfg.cudnn.enabled
    #model
    model = HRCapsModel(cnfg)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
    #print(summary(model,input_size=(3,224,224)))  
    # for param in model.parameters():
    #     if param.requires_grad == False:
    #         print(param)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=cnfg.train.lr,
        momentum=cnfg.train.momentum,
        weight_decay=cnfg.train.wd,
        #dampening=cnfg.train.DAMPENING,
        nesterov=cnfg.train.nesterov
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if isinstance(cnfg.train.lr_step, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cnfg.train.lr_step, cnfg.train.lr_factor)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cnfg.train.lr_step, cnfg.train.lr_factor) 

    if cnfg.mode == 'train':
        store_dir = os.path.join('results', datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
        os.mkdir(store_dir)
        return model, optimizer, criterion, lr_scheduler, store_dir
    else:
        return model, criterion

#Saved Model
def saved_model(model, optimizer=''):
    best_frame_auc = 0  # best validate accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    if cnfg.resume_dir:
        print('==> Resuming from checkpoint..')
        if(cnfg.mode == 'train'):
            print("Training....")
            #checkpoint = torch.load(os.path.join(cnfg.resume_dir, 'after_ckpt.pth'))
            #checkpoint = torch.load(os.path.join(cnfg.resume_dir, 'best_ckpt.pth'))
            #checkpoint = torch.load(cnfg.resume_dir)
            checkpoint = torch.load(os.path.join(cnfg.resume_dir, 'train-16.pth'))
        else:
            print("Testing....")
            #checkpoint = torch.load(os.path.join(cnfg.resume_dir, 'best_ckpt.pth'))
            #checkpoint = torch.load(os.path.join(cnfg.resume_dir, 'after_ckpt.pth'))
            #checkpoint = torch.load(cnfg.resume_dir)
            checkpoint = torch.load(os.path.join(cnfg.resume_dir, 'train-16.pth'))
        model.load_state_dict(checkpoint['model'])
        best_frame_auc = checkpoint['frame_level_auc']
        #best_frame_auc = 0.847
        start_epoch = checkpoint['epoch']
        if(cnfg.mode == 'train'):
            optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, best_frame_auc, start_epoch

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def load_data():
    train_dir = cnfg.data_dir + '/train' # training_set contains training dataset
    validate_dir = cnfg.data_dir + '/val'  #contains validate dataset
    test_dir =cnfg.data_dir + '/test' # training_set contains training dataset
    num_class = len(next(os.walk(cnfg.data_dir))[1])

    transformation = transforms.ToTensor()

    if cnfg.mode == 'train':
        #trainSet = datasets.ImageFolder(train_dir, transform = transformation)

        #trainMean, trainStd = get_mean_and_std(trainSet)

        train_transformation = transforms.Compose([
                                transforms.Resize((300,300)),
                                transforms.RandomCrop(cnfg.image_size),
                                #transforms.RandomChoice([
                                transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.2),
                                #transforms.RandomAffine(degrees, translate=None, scale=None,shear=None, resample=False, fillcolor=0)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(180),
                                #]),
                                transforms.ToTensor(),
                                #transforms.Normalize((trainMean[0],trainMean[1],trainMean[2]),(trainStd[0],trainStd[1],trainStd[2]))
                                transforms.Normalize((0.4674, 0.3416, 0.3141),
                                                    (0.1940, 0.1671, 0.1574))
                            ])
        val_transformation = transforms.Compose([
                                transforms.Resize((cnfg.image_size,cnfg.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4674, 0.3416, 0.3141),
                                                    (0.1940, 0.1671, 0.1574))
                            ])
        #Load the dataset with Image Folder
        train_set = ImageFolderWithPaths(train_dir, transform = train_transformation)
        val_set = ImageFolderWithPaths(validate_dir, transform = val_transformation)
        #define data loaders
        train_loader = DataLoader(train_set, batch_size=cnfg.batch_size,shuffle= True,num_workers=cnfg.num_workers)
        val_loader = DataLoader(val_set, batch_size=cnfg.batch_size,num_workers=cnfg.num_workers)
        return num_class, train_loader, val_loader
    else:
        test_transformation = transforms.Compose([
                                transforms.Resize((cnfg.image_size,cnfg.image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4674, 0.3416, 0.3141),
                                                    (0.1940, 0.1671, 0.1574))
                            ])
        #test_set = datasets.ImageFolder(test_dir, transform = test_transformation)
        test_set = ImageFolderWithPaths(test_dir, transform = test_transformation)
        test_loader = DataLoader(test_set, batch_size=cnfg.batch_size,num_workers=cnfg.num_workers)
        return num_class, test_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_confusion_matrix(targets,preds, classes,epoch, mode, normalize=False,  cmap=plt.cm.Blues):
    num_classes = len(classes)
    stacked = torch.stack((targets,preds),dim=1)
    #print(stacked)
    cmt = torch.zeros(num_classes,num_classes, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        #print(p)
        cmt[tl, pl] += 1
    cm = cmt.numpy()

    targets = targets.to('cpu').numpy()
    preds = preds.to('cpu').numpy()
    tn, fp, fn, tp = confusion_matrix(targets,preds).ravel()
    print(f"frame \n TN = {tn}\nFP = {fp}\nFN = {fn}\nTP = {tp}\n")
    #recall = float(cm[1,1])/(cm[1,1]+cm[0,1]) * 100.0
    recall = float(tp/(tp+fn)) * 100.0
    #precision = float(cm[1,1])/(cm[1,1]+cm[1,0]) * 100.0
    precision = float(tp)/(tp+fp) * 100.0
    f1 = (2*recall*precision)/(recall+precision)
    #alpha = 100 # from DFDC Initial Dataset Paper
    #weighted_percision = float(cm[0,0])/(cm[0,0]+(alpha*cm[1,0]))
    #log_wP_base10 = math.log10(weighted_percision)
    #log_wP_base2 = math.log2(weighted_percision)
    if mode == "train":
        tb.add_scalar('train_precision', precision, epoch)
        tb.add_scalar('train_recall', recall, epoch)
        tb.add_scalar('train_f1', f1, epoch)
        #tb.add_scalar('train_logwP', log_wP_base10, epoch)
    elif mode == "validate":
        tb.add_scalar('validate_precision', precision, epoch)
        tb.add_scalar('validate_recall', recall, epoch)
        tb.add_scalar('validate_f1', f1, epoch)
        #tb.add_scalar('validate_logwP', log_wP_base10, epoch)
    else:
        tb.add_scalar('test_precision', precision, epoch)
        tb.add_scalar('test_recall', recall, epoch)
        tb.add_scalar('test_f1', f1, epoch)
        #tb.add_scalar('test_logwP', log_wP_base10, epoch)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(f"Frame Level Confusion Matrix -----> \n{cm}") 
    #print(cm)
    #print(f"{mode} ---> frame level -> Percision = {precision:.3f}% \t Recall = {recall:.3f}%  \tF1 = {f1:.3f}%")
    #print(f"{mode} ---> log(Weighted Percision) = {log_wP_base10} \t 0 being is the maximum achievable")
    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(mode + " confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tb.add_figure(mode + " confusion matrix", fig,epoch)
    return precision,recall,f1,cm

def plot_video_cm(video_cm, mode, epoch, normalize=False,  cmap=plt.cm.Blues):
    classes = ["real","fake"]
    fig = plt.figure(figsize=(8,8))
    plt.imshow(video_cm, interpolation='nearest', cmap=cmap)
    plt.title(mode + " video confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = video_cm.max() / 2.
    for i, j in itertools.product(range(video_cm.shape[0]), range(video_cm.shape[1])):
        plt.text(j, i, format(video_cm[i, j], fmt), horizontalalignment="center", color="white" if video_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tb.add_figure(mode + " video confusion matrix", fig,epoch)

def video_level_acc(all_frames_names, all_targets, all_preds,mode,epoch):
    video_cm = np.array([[0,0], [0,0]])
    correct = 0
    videos = dict()
    vid_name = []
    targets = all_targets.to('cpu').numpy()
    preds = all_preds.to('cpu').numpy()

    frame_level_auc = roc_auc_score(targets,preds)
    
    # with open(f"{mode}_{epoch}_mis_frame.csv", 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["frame_name", "target", "pred"])
    #     for i in range(len(all_frames_names)):
    #         if int(targets[i]) != int(preds[i]):
    #             writer.writerow([all_frames_names[i], int(targets[i]), int(preds[i])])


    with open('celeb_reface_frame_acc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_name", "target", "pred"])
        for i in range(len(all_frames_names)):
            writer.writerow([all_frames_names[i], int(targets[i]), int(preds[i])])


    #print(len(all_frames_names))
    for video in all_frames_names:  #Get video name from frames names
        #vid_name.append(video.split('/')[-1].split('-')[2])    #yolo_histEq
        #vid_name.append(video.split('/')[-1].split('-')[2])   #mtcnn
        vid_name.append(video.split('/')[-1].split('-')[1]) #dfdc_faces (%3 & all) (train & test)
        #vid_name.append(video.split('/')[-1].split('-')[2]) #yolo_2M_actor_128 (%3 & all) (train & test)
    #print(len(vid_name))
    for i in range(len(vid_name)):
        video_name = vid_name[i]
        if video_name in videos:
            videos[video_name]["sum_pred"] += int(preds[i])
            videos[video_name]["frames_count"] += 1
        else:
            videos[video_name]= {"frames_count":1,"sum_pred":int(preds[i]),"target":int(targets[i]),"avg":0, "pred":0}
    #print(len(videos))
    for vid in videos:
        videos[vid]["avg"] = float(videos[vid]["sum_pred"])/videos[vid]["frames_count"]
        if videos[vid]["avg"] >= 0.5:
            videos[vid]["pred"] = 1
        else:
            videos[vid]["pred"] = 0
        video_cm[videos[vid]["target"],videos[vid]["pred"]] += 1
        if videos[vid]["pred"] == videos[vid]["target"]:
            correct += 1

    vid_avg_preds = []
    vid_targets = []
    vid_preds = []
    for vid in videos:
        vid_avg_preds.append(videos[vid]["avg"])
        vid_targets.append(videos[vid]["target"])
        vid_preds.append(videos[vid]["pred"])

    video_level_auc = roc_auc_score(vid_targets,vid_avg_preds)
    tn, fp, fn, tp = confusion_matrix(vid_targets,vid_preds).ravel()
    print(f"video \n TN = {tn}\nFP = {fp}\nFN = {fn}\nTP = {tp}\n")

    videos_names = []
    for key in videos.keys(): 
        videos_names.append(key) 

    with open('celeb_reface_video_acc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["video_name", "target", "avg_pred", "pred"])
        for i in range(len(vid_avg_preds)):
            writer.writerow([videos_names[i], int(vid_targets[i]), float(vid_avg_preds[i]),int(vid_preds[i])])


    # wb = Workbook() # Workbook is created 
    # video_sheet = wb.add_sheet('video_level')    # add_sheet is used to create sheet. 
    # video_sheet.write(0,0,"video_name")
    # video_sheet.write(0,1,"target")
    # video_sheet.write(0,2,"pred")
    # for i in range(len(vid_avg_preds)):
    #     video_sheet.write(i+1,0,vid_name[i])
    #     video_sheet.write(i+1,1,int(vid_targets[i]))
    #     video_sheet.write(i+1,2,float(vid_avg_preds[i]))
    # wb.save(f"{mode}-target-pred.xls") 

    all_vid_preds = torch.tensor(vid_avg_preds, dtype=float)
    all_vid_targets = torch.tensor(vid_targets,dtype=int)

    # metric = PrecisionRecall()
    # precision, recall, thresholds = metric(all_vid_preds, all_vid_targets)

    #print("Number of Precision values = ",len(precision))
    #print(precision)
    #f = open("precision_values.txt","w")
    #f.write(str(precision))
    #f.close()
    #print("Number of Recall values = ",len(recall))
    #print(recall)
    #f = open("recall_values.txt","w")
    #f.write(str(recall))
    #f.close()
    #print("Number of Thresholds values = ",len(thresholds))
    #print(thresholds)
    #f = open("thresholds_values.txt","w")
    #f.write(str(thresholds))
    #f.close()

    # closest_zero = np.argmin(np.abs(thresholds))
    # closest_zero_p = precision[closest_zero]
    # closest_zero_r = recall[closest_zero]

    # #plt.figure()
    # fig = plt.figure(figsize=(8,8))
    # plt.xlim([0.0, 1.01])
    # plt.ylim([0.0, 1.01])
    # plt.plot(recall, precision, label='Recall-Precision Curve')
    # plt.plot(closest_zero_r, closest_zero_p, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    # plt.xlabel('Recall', fontsize=16)
    # plt.ylabel('Precision', fontsize=16)
    # #plt.axes().set_aspect('equal')
    # #plt.show()
    # plt.tight_layout()
    # tb.add_figure(mode + " Recall-Precision Curve", fig,epoch)

    # print(f"Number of videos = {len(videos)}")
    # for pair in videos.items():
    #     print(pair)
    #json.dump(videos, open('test-videos.json','w'))
    #print(f"Video Level Confusion Matrix -----> \n{video_cm}") 

    alpha = 100 # from DFDC Initial Dataset Paper
    weighted_precision = float(tp)/(tp+(alpha*fp))
    #weighted_precision = float(video_cm[1,1])/(video_cm[1,1]+(alpha*video_cm[1,0]))
    log_wP_natural = np.log(weighted_precision)

    #video_recall = float(video_cm[1,1])/(video_cm[1,1]+video_cm[0,1]) * 100.0
    video_recall = float(tp)/(tp+fn) * 100.0
    #video_precision = float(video_cm[1,1])/(video_cm[1,1]+video_cm[1,0]) * 100.0
    video_precision = float(tp)/(tp+fp) * 100.0
    video_f1 = (2*video_recall*video_precision)/(video_recall+video_precision) 

    #print(f"{mode} ---> log10(Weighted Precision) = {log_wP_base10} \t 0 being is the maximum achievable")
    #print(f"{mode} --> video level ---> log(Weighted Precision) = {log_wP_natural} \t 0 being is the maximum achievable")
    if mode == "train":
        tb.add_scalar('train_logwP', log_wP_natural, epoch)
    elif mode == "validate":
        tb.add_scalar('validate_logwP', log_wP_natural, epoch)
    else:
        tb.add_scalar('test_logwP', log_wP_natural, epoch)
    plot_video_cm(video_cm,mode,epoch)
    vid_acc = 100 * float(correct)/len(videos)
    return vid_acc, log_wP_natural, video_recall, video_precision, video_f1, video_cm,video_level_auc, frame_level_auc

def model_in_action(mode,loader,epoch):
    global best_frame_auc
    mode_loss = 0
    correct = 0
    total = 0
    all_preds = torch.tensor([], dtype=int, device=device)
    all_targets = torch.tensor([],dtype=int,device=device)
    all_frames_names = []

    print(f"<---------------{mode}--------------->")
    if mode == 'train':
        # switch to train mode
        model.train()
        for batch_idx, (inputs, targets,path) in enumerate(loader):
            # measure data loading time
            #data_time.update(time.time() - end)

            for frame in path:
                all_frames_names.append(frame)

            inputs = inputs.to(device)

            targets = targets.to(device)
            #print(targets)
            all_targets = torch.cat((all_targets, targets),dim=0)

            # compute output
            output = model(inputs,path)
            loss = criterion(output, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mode_loss += loss.item()
            _, predicted = output.max(dim=1)
            #print(predicted)
            all_preds = torch.cat((all_preds, predicted),dim=0)

            total += targets.size(0)
                
            correct += predicted.eq(targets).sum().item()

            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()

            progress_bar(batch_idx, len(loader), 
                        f"Loss: {mode_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% | {correct}/{total}")
    elif mode == "validate" or mode == "test":
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(loader):
                for frame in path:
                    all_frames_names.append(frame)

                inputs = inputs.to(device)

                targets = targets.to(device)
                all_targets = torch.cat((all_targets, targets),dim=0)
                
                # compute output
                output = model(inputs,path)
                loss = criterion(output, targets)
                
                mode_loss += loss.item()
                
                _, predicted = output.max(dim=1)
                all_preds = torch.cat((all_preds, predicted),dim=0)
                    
                total += targets.size(0)
                
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(loader), 
                        f"Loss: {mode_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% | {correct}/{total}")

    fr_precision,fr_recall,fr_f1,fr_cm = plot_confusion_matrix(all_targets,all_preds,['real','fake'],epoch,mode)

    epoch_acc = 100.*correct/total
    epoch_loss = mode_loss/len(loader)

    vid_acc, vid_log_wP_natural, vid_recall, vid_precision, vid_f1, vid_cm,video_level_auc, frame_level_auc = video_level_acc(all_frames_names, all_targets, all_preds,mode,epoch)
    #torch.cuda.empty_cache()
    if mode == "validate":
        if frame_level_auc > best_frame_auc:
            print('Saving..')
            state = {
                    'model': model.state_dict(),
                    'acc': epoch_acc,
                    'frame_level_auc':frame_level_auc,
                    'epoch': epoch+1,
                    'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(store_dir, 'best_ckpt.pth'))
            best_frame_auc = frame_level_auc

        state = {
                'model': model.state_dict(),
                'acc': epoch_acc,
                'frame_level_auc':frame_level_auc,
                'epoch': epoch+1,
                'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(store_dir, f"train-{epoch}.pth"))
    
    return epoch_acc, epoch_loss, vid_acc, vid_log_wP_natural, vid_recall, vid_precision, vid_f1, fr_precision,fr_recall,fr_f1, fr_cm, vid_cm,video_level_auc, frame_level_auc 

def main():
    print(f"Starting time : {datetime.now()}")
    configure()
    global device, num_class, train_loader, validate_loader, test_loader
    global model, optimizer, criterion, lr_scheduler, store_dir, best_frame_auc, start_epoch
    global tb
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_number = 4
    if cnfg.mode == "train":
        tb = SummaryWriter(f"runs/tr-{run_number}-{cnfg.data_dir.split('/')[-1]}")
    else:    
        tb = SummaryWriter(f"runs/ts-{run_number}-{cnfg.data_dir.split('/')[-1]}")

    if cnfg.mode == 'train':
        num_class, train_loader, validate_loader = load_data()
        model, optimizer, criterion, lr_scheduler, store_dir = model_build()
        model, optimizer, best_frame_auc, start_epoch = saved_model(model, optimizer)
    else:
        num_class, test_loader = load_data()
        model, criterion = model_build()
        model, optimizer, best_frame_auc, start_epoch = saved_model(model)

    #tb.add_graph(model)

    print(f"Model best validation frame auc = {best_frame_auc}")
    print(f"Number of model parameters: {count_parameters(model)}")

    if cnfg.mode == 'train':
        results = {
        #'best_acc':0,
        'total_epochs':cnfg.epochs,
        'batch_size': cnfg.batch_size,
        'data_dir':cnfg.data_dir,
        'resume_dir':cnfg.resume_dir,
        #'learning rate':cnfg.train.lr,
        #'weight_decay':cnfg.train.wd,
        'train_validate_acc': [],
        'train_validate_loss': [],
        'train_validate_vid_acc':[],
        'train_validate_log_wP':[],
        'train_validate_frame_precision':[],
        'train_validate_frame_recall':[],
        'train_validate_frame_f1':[],
        'train_validate_vid_precision':[],
        'train_validate_vid_recall':[],
        'train_validate_vid_f1':[]
        }

        for epoch in range(start_epoch, start_epoch+cnfg.epochs):
            print(f"\nEpoch: {epoch}")
            tr_acc, tr_loss, tr_vid_acc, tr_vid_log_wP, tr_vid_rec, tr_vid_pr, tr_vid_f1, tr_fr_pr,tr_fr_rec,tr_fr_f1, tr_fr_cm, tr_vid_cm,tr_vid_auc, tr_fr_auc = model_in_action("train",train_loader,epoch)
            lr_scheduler.step()
            vl_acc, vl_loss, vl_vid_acc, vl_vid_log_wP, vl_vid_rec, vl_vid_pr, vl_vid_f1, vl_fr_pr,vl_fr_rec,vl_fr_f1, vl_fr_cm, vl_vid_cm,vl_vid_auc, vl_fr_auc = model_in_action("validate",validate_loader,epoch)
            
            print("***************Training Results***************")
            print(f"Frame Level Confusion Matrix -----> \n{tr_fr_cm}")
            print(f"Video Level Confusion Matrix -----> \n{tr_vid_cm}")
            print(f"frame_level_acc = {tr_acc:.3f}% \t \t video_level_acc = {tr_vid_acc:.3f}%")
            print(f"frame_level_precision = {tr_fr_pr:.3f}% \t video_level_precision = {tr_vid_pr:.3f}%")
            print(f"frame_level_recall = {tr_fr_rec:.3f}% \t \t video_level_recall = {tr_vid_rec:.3f}%")
            print(f"frame_level_f1 = {tr_fr_f1:.3f}% \t \t video_level_f1 = {tr_vid_f1:.3f}%")
            print(f"frame_level_roc_auc = {tr_fr_auc:.3f} \t \t video_level_roc_auc = {tr_vid_auc:.3f}")
            print(f"frame_level_loss = {tr_loss:.3f}")
            print(f"video_level_Log(wP) = {tr_vid_log_wP:.3f} \t \t 0 is the maximum achievable value")
            print("***************Validation Results***************")
            print(f"Frame Level Confusion Matrix -----> \n{vl_fr_cm}")
            print(f"Video Level Confusion Matrix -----> \n{vl_vid_cm}")
            print(f"frame_level_acc = {vl_acc:.3f}% \t \t video_level_acc = {vl_vid_acc:.3f}%")
            print(f"frame_level_precision = {vl_fr_pr:.3f}% \t video_level_precision = {vl_vid_pr:.3f}%")
            print(f"frame_level_recall = {vl_fr_rec:.3f}% \t \t video_level_recall = {vl_vid_rec:.3f}%")
            print(f"frame_level_f1 = {vl_fr_f1:.3f}% \t \t video_level_f1 = {vl_vid_f1:.3f}%")
            print(f"frame_level_roc_auc = {vl_fr_auc:.3f} \t \t video_level_roc_auc = {vl_vid_auc:.3f}")
            print(f"frame_level_loss = {vl_loss:.3f}")
            print(f"video_level_Log(wP) = {vl_vid_log_wP:.3f} \t \t 0 is the maximum achievable value")

            #for saving data in a file
            results['train_validate_acc'].append([epoch, tr_acc, vl_acc])
            results['train_validate_loss'].append([epoch, tr_loss, vl_loss])
            results['train_validate_vid_acc'].append([epoch, tr_vid_acc, vl_vid_acc])
            results['train_validate_log_wP'].append([epoch, tr_vid_log_wP, vl_vid_log_wP])
            results['train_validate_frame_precision'].append([epoch, tr_fr_pr, vl_fr_pr])
            results['train_validate_frame_recall'].append([epoch, tr_fr_rec, vl_fr_rec])
            results['train_validate_frame_f1'].append([epoch, tr_fr_f1, vl_fr_f1])
            results['train_validate_vid_precision'].append([epoch, tr_vid_pr, vl_vid_pr])
            results['train_validate_vid_recall'].append([epoch, tr_vid_rec, vl_vid_rec])
            results['train_validate_vid_f1'].append([epoch, tr_vid_f1, vl_vid_f1])

            #for drawing lines
            tb.add_scalar('train_loss', tr_loss, epoch)
            tb.add_scalar('train_accuracy', tr_acc, epoch)
            tb.add_scalar('validate_loss', vl_loss, epoch)
            tb.add_scalar('validate_accuracy', vl_acc, epoch)
            tb.add_scalars('train_validate_frame_acc',{'train_acc':tr_acc,'validate_acc':vl_acc},epoch)
            tb.add_scalars('train_validate_frame_loss',{'train_loss':tr_loss,'validate_loss':vl_loss},epoch)
            tb.add_scalars('train_validate_vid_acc',{'train_vid_acc':tr_vid_acc,'validate_vid_acc':vl_vid_acc},epoch)
            tb.add_scalars('train_validate_vid_precision',{'train_vid_pr':tr_vid_pr,'validate_vid_pr':vl_vid_pr},epoch)
            tb.add_scalars('train_validate_vid_recall',{'train_vid_rec':tr_vid_rec,'validate_vid_rec':vl_vid_rec},epoch)
            
        results['best_frame_auc']=best_frame_auc
        tb.close()

        #Writing in files
        store_file = os.path.join(store_dir, f"{cnfg.epochs}--{run_number}.dict")
        store_file2 = os.path.join(store_dir, f"{cnfg.epochs}--{ cnfg.image_size}.json")

        json.dump(results, open(store_file2,'w'))
        pickle.dump(results, open(store_file, 'wb'))

        #Saving after last epoch
        state = {
                    'model': model.state_dict(),
                    #'acc': best_acc,
                    'frame_level_auc':best_frame_auc,
                    'epoch': start_epoch+cnfg.epochs,
                    'optimizer': optimizer.state_dict(),
                }
        torch.save(state, os.path.join(store_dir, 'after_ckpt.pth'))
    else:
        ts_acc, ts_loss, ts_vid_acc, ts_vid_log_wP, ts_vid_rec, ts_vid_pr, ts_vid_f1, ts_fr_pr,ts_fr_rec,ts_fr_f1, ts_fr_cm, ts_vid_cm, ts_vid_auc, ts_fr_auc = model_in_action("test",test_loader,0)
        print(f"model_dir -----> aug - {cnfg.resume_dir}")
        print("***************Test Results***************")
        print(f"Frame Level Confusion Matrix -----> \n{ts_fr_cm}")
        print(f"Video Level Confusion Matrix -----> \n{ts_vid_cm}")
        print(f"frame_level_acc = {ts_acc:.3f}% \t \t video_level_acc = {ts_vid_acc:.3f}%")
        print(f"frame_level_precision = {ts_fr_pr:.3f}% \t video_level_precision = {ts_vid_pr:.3f}%")
        print(f"frame_level_recall = {ts_fr_rec:.3f}% \t \t video_level_recall = {ts_vid_rec:.3f}%")
        print(f"frame_level_f1 = {ts_fr_f1:.3f}% \t \t video_level_f1 = {ts_vid_f1:.3f}%")
        print(f"frame_level_roc_auc = {ts_fr_auc:.3f} \t \t video_level_roc_auc = {ts_vid_auc:.3f}")
        print(f"frame_level_loss = {ts_loss:.3f}")
        print(f"video_level_Log(wP) = {ts_vid_log_wP} \t \t 0 is the maximum achievable value")
        tb.close()

    print(f"Ending time : {datetime.now()}")

if __name__ == '__main__':
    main()
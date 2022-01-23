'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import numpy as np, json, argparse, time
from numpy.core.fromnumeric import size
from tqdm import tqdm
import cv2, logging
from pathlib import Path
import torch, torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, confusion_matrix, accuracy_score
from supervised.apply.utils import *
import bc_config

class Train_Util:

    def __init__(self, experiment_description, epochs, model, device, train_loader, val_loader, optimizer, criterion, batch_size, scheduler, num_classes, writer, threshold = 0.2):
        self.experiment_description = experiment_description
        self.epochs = epochs
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.writer = writer
        self.num_classes = num_classes
        self.threshold = threshold
        

    def train_epoch(self):
        self.model.train()
        loss_agg = Aggregator()
        confusion_matrix_epoch = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        with tqdm(total=(len(self.train_loader))) as (t):
            for magnification, item_dict, binary_label, multi_label in tqdm(self.train_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)                
                
                
                target = binary_label.to(self.device)

                outputs = self.model(view)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)
                loss = self.criterion(outputs, target)
                
                
                
                predicted = (outputs > self.threshold).int()

                predicted = predicted.to(self.device)

                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_epoch[(targetx.long(), predictedx.long())] += 1
               
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_agg.update(loss.item())
                t.set_postfix(loss=('{:05.3f}'.format(loss_agg())))
                t.update()

        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_epoch)
        print(f'{self.experiment_description}:classwise precision', classwise_precision)
        print(f'{self.experiment_description}: classwise recall', classwise_recall)
        print(f'{self.experiment_description}: classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Accuracy', accuracy)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, loss_agg())
    
    def evaluate_validation_set(self):
        confusion_matrix_val = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        self.model.eval()
        val_loss_avg = Aggregator()
        with torch.no_grad():
            for magnification, item_dict, binary_label, multi_label in tqdm(self.val_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)                
                
                
                target = binary_label.to(self.device)

                outputs = self.model(view)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)
                loss = self.criterion(outputs, target)
         
                predicted = (outputs > self.threshold).int()

                predicted = predicted.to(self.device)

                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                else:
                    val_loss_avg.update(loss.item())

        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        
        print(f'{self.experiment_description}: Validation classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Validation classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Validation classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Validation Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Validation Accuracy', accuracy)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, val_loss_avg())

    
    def test_model(self):
        confusion_matrix_val = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        self.model.eval()
        val_loss_avg = Aggregator()
        with torch.no_grad():
            for magnification, item_dict, binary_label, multi_label in tqdm(self.val_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)                
                
                
                target = binary_label.to(self.device)

                outputs = self.model(view)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)

                
                predicted = (outputs > self.threshold).int()

                predicted = predicted.to(self.device)

                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1

        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        
        print(f'{self.experiment_description}: Test classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Test classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Test classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Test Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Test Accuracy', accuracy)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1)
    
    
    def test_class_probabilities(self, model, device, test_loader, which_class):
        model.Test()
        actuals = []
        probabilities = []
        Testth torch.no_grad():
            for Testage, label in test_loader:
                image, label = image.to(device), label.to(device)
                output = torch.sigmoid(model(image))
                prediction = output.argmax(dim=1, keepdim=True)
                actuals.extend(label.view_as(prediction) == which_class)
                output = output.cpu()
                probabilities.extend(np.exp(output[:, which_class]))

        return (
         [i.item() for i in actuals], [i.item() for i in probabilities])

    def train_and_evaluate(self):
        
        best_f1 = 0.0
        for epoch in range(self.epochs):
            #train epoch
            weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1, loss = self.train_epoch()
            #evaluate on validation set
            val_weighted_f1, val_accuracy, val_classwise_precision,val_classwise_recall,val_classwise_f1, val_loss = self.evaluate_validation_set()
                        
            print("Epoch {}/{} Train Loss:{}, Val Loss: {}".format(epoch, self.epochs, loss, val_loss))

            if best_f1 < val_weighted_f1:
                best_f1 = val_weighted_f1
                result_path = f"{bc_config.result_path}{self.experiment_description}"
                Path(result_path).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), f"{result_path}/_{epoch}_{val_weighted_f1}.pth")
            
            self.scheduler.step(val_loss)

            #Tensorboard
            self.writer.add_scalar('Loss/Validation_Set', val_loss, epoch)
            self.writer.add_scalar('Loss/Training_Set', loss, epoch)

            self.writer.add_scalar('Accuracy/Validation_Set', val_accuracy, epoch)
            self.writer.add_scalar('Accuracy/Training_Set', accuracy, epoch)
            
            self.writer.add_scalar('Weighted F1/Validation_Set', val_weighted_f1, epoch)
            self.writer.add_scalar('Weighted F1/Training_Set', weighted_f1, epoch)
            
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            #Classwise metrics logging
            if 2 == self.num_classes:
                for index in range(0,len(bc_config.binary_label_list)):
                    
                    self.writer.add_scalar(f'F1/Validation_Set/{bc_config.binary_label_list[index]}', val_classwise_f1[index], epoch)
                    self.writer.add_scalar(f'F1/Training_Set/{bc_config.binary_label_list[index]}', classwise_f1[index], epoch)

                    self.writer.add_scalar(f'Precision/Validation_Set/{bc_config.binary_label_list[index]}', val_classwise_precision[index], epoch)
                    self.writer.add_scalar(f'Precision/Training_Set/{bc_config.binary_label_list[index]}', classwise_precision[index], epoch)

                    self.writer.add_scalar(f'Recall/Validation_Set/{bc_config.binary_label_list[index]}', val_classwise_recall[index], epoch)
                    self.writer.add_scalar(f'Recall/Training_Set/{bc_config.binary_label_list[index]}', classwise_recall[index], epoch)
                
    def process_classification_report(self, report):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split(' ')
            row_data = list(filter(None, row_data))
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            report_data.append(row)
        else:
            return report_data

    def get_metrics_from_confusion_matrix(self, confusion_matrix_epoch):
        epoch_classwise_precision_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu()) / np.array(confusion_matrix_epoch.cpu()).sum(axis=0)
        epoch_classwise_precision_manual_cpu = np.nan_to_num(epoch_classwise_precision_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_classwise_recall_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu()) / np.array(confusion_matrix_epoch.cpu()).sum(axis=1)
        epoch_classwise_recall_manual_cpu = np.nan_to_num(epoch_classwise_recall_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_classwise_f1_manual_cpu = 2 * (epoch_classwise_precision_manual_cpu * epoch_classwise_recall_manual_cpu) / (epoch_classwise_precision_manual_cpu + epoch_classwise_recall_manual_cpu)
        epoch_classwise_f1_manual_cpu = np.nan_to_num(epoch_classwise_f1_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_avg_f1_manual = np.sum(epoch_classwise_f1_manual_cpu * np.array(confusion_matrix_epoch.cpu()).sum(axis=1)) / np.array(confusion_matrix_epoch.cpu()).sum(axis=1).sum()
        epoch_acc_manual = 100 * np.sum(np.array(confusion_matrix_epoch.diag().cpu())) / np.sum(np.array(confusion_matrix_epoch.cpu()))
        return (
         epoch_avg_f1_manual, epoch_acc_manual, epoch_classwise_precision_manual_cpu, epoch_classwise_recall_manual_cpu, epoch_classwise_f1_manual_cpu)

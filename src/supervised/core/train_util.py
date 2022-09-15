'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

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
from supervised.apply.sampler import BalancedBatchSampler

from supervised.apply.utils import *
import bc_config

def BCELoss_ClassWeights(input, target, class_weights):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - target * torch.log(input) - (1 - target) * torch.log(1 - input)
        weighted_bce = (bce * class_weights).sum(axis=1) / class_weights.sum(axis=1)[0]
        final_reduced_over_batch = weighted_bce.mean(axis=0)
        return final_reduced_over_batch

class Train_Util:

    def __init__(self, experiment_description, epochs, model, device, train_loader, val_loader, optimizer, criterion, batch_size, scheduler, num_classes, writer, early_stopping_patience, test_loader = None, batch_balancing=False, threshold = 0.2, result_folder= 'default/', linear_eval = False):
        self.experiment_description = experiment_description
        self.epochs = epochs
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.writer = writer
        self.num_classes = num_classes
        self.threshold = threshold
        self.early_stopping_patience = early_stopping_patience
        self.batch_balancing = batch_balancing
        self.result_folder = result_folder
        self.LE = linear_eval

    def get_weights(self, target):
        pos_weight = torch.sum(target, dim=0)/self.batch_size
        neg_weight = 1 - pos_weight
        pos_weight = 1/pos_weight
        neg_weight = 1/neg_weight
        target_array = target.cpu().numpy()
        target_array[target_array == 1.0] = pos_weight.cpu().numpy()
        target_array[target_array == 0.0] = neg_weight.cpu().numpy()
        weights = torch.tensor(target_array)
        weights = weights.to(self.device)
        return weights
    
    def train_epoch(self):
        self.model.train()
        
        loss_agg = Aggregator()
        confusion_matrix_epoch = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        with tqdm(total=(len(self.train_loader))) as (t):
            for patient_id, magnification, item_dict, binary_label, multi_label in tqdm(self.train_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)                
                target = binary_label.to(self.device)
                outputs = self.model(view)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)
                if True == self.batch_balancing:
                    self.criterion = torch.nn.BCELoss(weight=self.get_weights(target))
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
        print(confusion_matrix_epoch)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, loss_agg())
    
    def evaluate_validation_set(self):
        confusion_matrix_val = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        patient_id_dict_correct = {}
        patient_id_dict_total = {}
        self.model.eval()
        val_loss_avg = Aggregator()
        with torch.no_grad():
            for patient_id, magnification, item_dict, binary_label, multi_label in tqdm(self.val_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)                
                target = binary_label.to(self.device)
                outputs = self.model(view)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)
                if True == self.batch_balancing:
                    self.criterion = torch.nn.BCELoss(weight=self.get_weights(target))
                loss = self.criterion(outputs, target)
                predicted = (outputs > self.threshold).int()
                # for patient level accuracy
                for i in range(len(patient_id)):
                    if patient_id[i] in patient_id_dict_total.keys():
                        patient_id_dict_total[patient_id[i]] += 1
                    else:
                        patient_id_dict_total[patient_id[i]] = 1
                    if binary_label[i].item() == predicted[i].item():
                        if patient_id[i] in patient_id_dict_correct.keys():
                            patient_id_dict_correct[patient_id[i]] += 1
                        else:
                            patient_id_dict_correct[patient_id[i]] = 1
                predicted = predicted.to(self.device)
                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                else:
                    val_loss_avg.update(loss.item())

        total_patient_score = 0.0
        for key in patient_id_dict_total.keys():
            correct = 0
            if key in patient_id_dict_correct.keys():
                correct = patient_id_dict_correct[key]
            total_patient_score += correct/patient_id_dict_total[key]
        patient_level_accuracy = 100*(total_patient_score/len(patient_id_dict_total))
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        print(f'{self.experiment_description}: Validation classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Validation classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Validation classwise f1', classwise_f1)
        #print(f'{self.experiment_description}: Validation MCC', mcc)
        print(f'{self.experiment_description}: Validation Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Validation Accuracy', accuracy)
        print(f'{self.experiment_description}: Validation Patient-Level Accuracy', patient_level_accuracy)
        print(confusion_matrix_val)
        return (weighted_f1, accuracy, patient_level_accuracy, classwise_precision, classwise_recall, classwise_f1, val_loss_avg())
    
    def train_epoch_bach(self):
        # data strctures for image level classification from patches - majority voting
        example_total_dict = {}
        example_pred_total = {}
        example_gt_total = {}
        self.model.train()
        loss_agg = Aggregator()
        confusion_matrix_epoch = torch.zeros(len(bc_config.bach_label_list), len(bc_config.bach_label_list))
        confusion_matrix_epoch_image_level = torch.zeros(len(bc_config.bach_label_list), len(bc_config.bach_label_list))
        with tqdm(total=(len(self.train_loader))) as (t):
            for example_id, part_id, item_dict, multi_label in tqdm(self.train_loader):
                view = item_dict["image"]
                view = view.cuda(self.device, non_blocking=True)
                multi_label = multi_label.type(torch.LongTensor)                
                target = multi_label.to(self.device)
                outputs = self.model(view)
                _, predicted = torch.max(outputs, 1)
                loss = self.criterion(outputs, target)
                predicted = predicted.to(self.device)
                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_epoch[(targetx.long(), predictedx.long())] += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_agg.update(loss.item())
                t.set_postfix(loss=('{:05.3f}'.format(loss_agg())))
                t.update()
                #record keeping for majority voting over patches
                for idx in range(0, len(example_id)):
                    example = example_id[idx]
                    ground_truth = multi_label[idx]
                    predicted_label = predicted[idx].cpu().item()
                    example_gt_total[example] = ground_truth
                    if example in list(example_total_dict.keys()):
                        example_total_dict[example] += 1
                    else:
                        example_total_dict[example] = 1
                    if example in list(example_pred_total.keys()):
                        example_pred_total[example][predicted_label] += 1
                    else:
                        example_pred_total[example] = {0:0,1:0,2:0,3:0} #it can work for two or three classes also since we use max function
        image_wise_total, image_wise_correct = 0,0
        for example in list(example_total_dict.keys()):
            patch_wise_total = example_total_dict[example]
            patch_wise_gt = example_gt_total[example]
            patch_wise_pred_dict = example_pred_total[example]
            majority_voting_pred = max(patch_wise_pred_dict, key=patch_wise_pred_dict.get)
            confusion_matrix_epoch_image_level[patch_wise_gt, majority_voting_pred] += 1
            image_wise_total += 1
            if patch_wise_gt == majority_voting_pred:
                image_wise_correct += 1
        image_wise_accuracy = ((image_wise_correct*100)/image_wise_total)
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_epoch)
        weighted_f1_image_level, accuracy_image_level, classwise_precision_image_level, classwise_recall_image_level, classwise_f1_image_level = self.get_metrics_from_confusion_matrix(confusion_matrix_epoch_image_level)
        print(f'{self.experiment_description}: Patch level classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Patch level classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Patch level classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Patch level F1', weighted_f1)
        print(f'{self.experiment_description}: Patch level Accuracy', accuracy)
        print('  ')
        print(f'{self.experiment_description}: Image level classwise precision', classwise_precision_image_level)
        print(f'{self.experiment_description}: Image level classwise recall', classwise_recall_image_level)
        print(f'{self.experiment_description}: Image level classwise f1', classwise_f1_image_level)
        print(f'{self.experiment_description}: Image level F1', weighted_f1_image_level)
        print(f'{self.experiment_description}: Image level Accuracy', image_wise_accuracy, accuracy_image_level)
        print(confusion_matrix_epoch_image_level)
        return (
            weighted_f1_image_level, 
                accuracy_image_level, 
                classwise_precision_image_level, 
                classwise_recall_image_level, 
                classwise_f1_image_level, 
                weighted_f1, 
                accuracy, 
                classwise_precision, 
                classwise_recall, 
                classwise_f1, 
                loss_agg()
                )
    def evaluate_validation_set_bach(self):
        # data strctures for image level classification from patches - majority voting
        example_total_dict = {}
        example_pred_total = {}
        example_gt_total = {}
        val_loss_avg = Aggregator()
        confusion_matrix_val = torch.zeros(len(bc_config.bach_label_list), len(bc_config.bach_label_list))
        confusion_matrix_val_image_level = torch.zeros(len(bc_config.bach_label_list), len(bc_config.bach_label_list))
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=(len(self.val_loader))) as (t):
                for example_id, part_id, item_dict, multi_label in tqdm(self.val_loader):
                    view = item_dict["image"]
                    view = view.cuda(self.device, non_blocking=True)
                    multi_label = multi_label.type(torch.LongTensor)                
                    target = multi_label.to(self.device)
                    outputs = self.model(view)
                    _, predicted = torch.max(outputs, 1)
                    loss = self.criterion(outputs, target)
                    predicted = predicted.to(self.device)
                    for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                        confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                    val_loss_avg.update(loss.item())
                    t.set_postfix(loss=('{:05.3f}'.format(val_loss_avg())))
                    t.update()
                    #record keeping for majority voting over patches
                    for idx in range(0, len(example_id)):
                        example = example_id[idx]
                        ground_truth = multi_label[idx]
                        predicted_label = predicted[idx].cpu().item()
                        example_gt_total[example] = ground_truth
                        if example in list(example_total_dict.keys()):
                            example_total_dict[example] += 1
                        else:
                            example_total_dict[example] = 1
                        if example in list(example_pred_total.keys()):
                            example_pred_total[example][predicted_label] += 1
                        else:
                            example_pred_total[example] = {0:0,1:0,2:0,3:0}

        image_wise_total, image_wise_correct = 0,0
        for example in list(example_total_dict.keys()):
            patch_wise_total = example_total_dict[example]
            patch_wise_gt = example_gt_total[example]
            patch_wise_pred_dict = example_pred_total[example]
            majority_voting_pred = max(patch_wise_pred_dict, key=patch_wise_pred_dict.get)
            confusion_matrix_val_image_level[patch_wise_gt, majority_voting_pred] += 1
            image_wise_total += 1
            if patch_wise_gt == majority_voting_pred:
                image_wise_correct += 1
        image_wise_accuracy = ((image_wise_correct*100)/image_wise_total)
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        weighted_f1_image_level, accuracy_image_level, classwise_precision_image_level, classwise_recall_image_level, classwise_f1_image_level = self.get_metrics_from_confusion_matrix(confusion_matrix_val_image_level)
        print(f'{self.experiment_description}: Validation classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Validation classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Validation classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Validation Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Validation Accuracy', accuracy)
        print('  ')
        print(f'{self.experiment_description}: Validation Image level classwise precision', classwise_precision_image_level)
        print(f'{self.experiment_description}: Validation Image level classwise recall', classwise_recall_image_level)
        print(f'{self.experiment_description}: Validation Image level classwise f1', classwise_f1_image_level)
        print(f'{self.experiment_description}: Validation Image level F1', weighted_f1_image_level)
        print(f'{self.experiment_description}: Validation Image level Accuracy', image_wise_accuracy, accuracy_image_level)
        print(confusion_matrix_val_image_level)
        return (weighted_f1_image_level, accuracy_image_level, classwise_precision_image_level, classwise_recall_image_level, classwise_f1_image_level, weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, val_loss_avg())
    
    def evaluate_testset_bach(self, model_weights_path = None):
        # data strctures for image level classification from patches - majority voting
        example_total_dict = {}
        example_pred_total = {}
        example_gt_total = {}
        confusion_matrix_val = torch.zeros(len(bc_config.bach_label_list), len(bc_config.bach_label_list))
        confusion_matrix_val_image_level = torch.zeros(len(bc_config.bach_label_list), len(bc_config.bach_label_list))
        # loading best model
        print(f"Start - loading weights for best performing model - {model_weights_path}")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        print(f"Stop - loading weights for best performing model - {model_weights_path}")
        self.model.eval()
        with torch.no_grad():
            with tqdm(total=(len(self.test_loader))) as (t):
                for example_id, part_id, item_dict, multi_label in tqdm(self.test_loader):
                    view = item_dict["image"]
                    view = view.cuda(self.device, non_blocking=True)
                    multi_label = multi_label.type(torch.LongTensor)              
                    target = multi_label.to(self.device)
                    outputs = self.model(view)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to(self.device)
                    for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                        confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                    #record keeping for majority voting over patches
                    for idx in range(0, len(example_id)):
                        example = example_id[idx]
                        ground_truth = multi_label[idx]
                        predicted_label = predicted[idx].cpu().item()
                        example_gt_total[example] = ground_truth
                        if example in list(example_total_dict.keys()):
                            example_total_dict[example] += 1
                        else:
                            example_total_dict[example] = 1
                        if example in list(example_pred_total.keys()):
                            example_pred_total[example][predicted_label] += 1
                        else:
                            example_pred_total[example] = {0:0,1:0,2:0,3:0}

        image_wise_total, image_wise_correct = 0,0
        for example in list(example_total_dict.keys()):
            patch_wise_total = example_total_dict[example]
            patch_wise_gt = example_gt_total[example]
            patch_wise_pred_dict = example_pred_total[example]
            majority_voting_pred = max(patch_wise_pred_dict, key=patch_wise_pred_dict.get)
            image_wise_total += 1
            confusion_matrix_val_image_level[patch_wise_gt, majority_voting_pred] += 1
            if patch_wise_gt == majority_voting_pred:
                image_wise_correct += 1
        image_wise_accuracy = ((image_wise_correct*100)/image_wise_total)
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        weighted_f1_image_level, accuracy_image_level, classwise_precision_image_level, classwise_recall_image_level, classwise_f1_image_level = self.get_metrics_from_confusion_matrix(confusion_matrix_val_image_level)
        print(f'{self.experiment_description}: Testset classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Testset classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Testset classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Testset Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Testset Accuracy', accuracy)
        print('  ')
        print(f'{self.experiment_description}: Testset Image level classwise precision', classwise_precision_image_level)
        print(f'{self.experiment_description}: Testset Image level classwise recall', classwise_recall_image_level)
        print(f'{self.experiment_description}: Testset Image level classwise f1', classwise_f1_image_level)
        print(f'{self.experiment_description}: Testset Image level F1', weighted_f1_image_level)
        print(f'{self.experiment_description}: Testset Image level Accuracy', image_wise_accuracy, accuracy_image_level)
        print(confusion_matrix_val_image_level)
        return (weighted_f1_image_level, accuracy_image_level, classwise_precision_image_level, classwise_recall_image_level, classwise_f1_image_level, weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1)
    
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
        
        print(f'{self.experiment_description}: Validation classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Validation classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Validation classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Validation Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Validation Accuracy', accuracy)
        print(confusion_matrix_val)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1)
    
    def test_class_probabilities(self, model, device, test_loader, which_class):
        model.eval()
        actuals = []
        probabilities = []
        with torch.no_grad():
            for image, label in test_loader:
                image, label = image.to(device), label.to(device)
                output = torch.sigmoid(model(image))
                prediction = output.argmax(dim=1, keepdim=True)
                actuals.extend(label.view_as(prediction) == which_class)
                output = output.cpu()
                probabilities.extend(np.exp(output[:, which_class]))
        return ([i.item() for i in actuals], [i.item() for i in probabilities])

    def train_and_evaluate(self):
        best_acc = 0.0
        best_patient_level_acc = 0.0
        best_f1 = 0.0
        best_classwise_precision = []
        best_classwise_recall = []
        best_classwise_f1 = []
        lowest_val_loss = 12345.0
        lowest_val_loss_epoch = 0
        for epoch in range(1,self.epochs + 1):
            #train epoch
            weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1, loss = self.train_epoch()
            #evaluate on validation set
            val_weighted_f1, val_accuracy, val_patient_level_accuracy, val_classwise_precision,val_classwise_recall,val_classwise_f1, val_loss = self.evaluate_validation_set()
            print("Epoch {}/{} Train Loss:{}, Val Loss: {}".format(epoch, self.epochs, loss, val_loss))
            if (epoch - lowest_val_loss_epoch) >= self.early_stopping_patience:
                print('Eearly stopping criteria matched - training stop at highest validation Accuracy, validation loss, and epoch', best_acc, val_loss, epoch)
                break
            if lowest_val_loss > val_loss:
                lowest_val_loss = val_loss
                lowest_val_loss_epoch = epoch        
            if best_acc < val_accuracy:
                best_acc = val_accuracy
                best_patient_level_acc = val_patient_level_accuracy
                best_f1 = val_weighted_f1
                best_classwise_precision = val_classwise_precision
                best_classwise_recall = val_classwise_recall
                best_classwise_f1 = val_classwise_f1
                result_path = f"{self.result_folder}/{self.experiment_description}"
                Path(result_path).mkdir(parents=True, exist_ok=True)
                for file in Path(result_path).glob('*'):
                    file.unlink()
                torch.save(self.model.state_dict(), f"{result_path}/_{epoch}_{val_accuracy}_{val_patient_level_accuracy}_{val_weighted_f1}.pth")
            
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
            elif 7 == self.num_classes:
                for index in range(0,len(bc_config.multi_label_list)):
                    self.writer.add_scalar(f'F1/Validation_Set/{bc_config.multi_label_list[index]}', val_classwise_f1[index], epoch)
                    self.writer.add_scalar(f'F1/Training_Set/{bc_config.multi_label_list[index]}', classwise_f1[index], epoch)
                    self.writer.add_scalar(f'Precision/Validation_Set/{bc_config.multi_label_list[index]}', val_classwise_precision[index], epoch)
                    self.writer.add_scalar(f'Precision/Training_Set/{bc_config.multi_label_list[index]}', classwise_precision[index], epoch)
                    self.writer.add_scalar(f'Recall/Validation_Set/{bc_config.multi_label_list[index]}', val_classwise_recall[index], epoch)
                    self.writer.add_scalar(f'Recall/Training_Set/{bc_config.multi_label_list[index]}', classwise_recall[index], epoch)
        return best_acc, best_patient_level_acc, best_f1, best_classwise_precision, best_classwise_recall, best_classwise_f1
               
    def train_and_evaluate_bach(self):
        best_acc_image_level = 0.0
        best_f1_image_level = 0.0
        best_classwise_precision_image_level = []
        best_classwise_recall_image_level = []
        best_classwise_f1_image_level = []
        best_acc = 0.0
        best_f1 = 0.0
        best_classwise_precision = []
        best_classwise_recall = []
        best_classwise_f1 = []
        best_model_weights_path = None
        #training
        train_best_acc_image_level = 0.0
        train_best_f1_image_level = 0.0
        train_best_classwise_precision_image_level = []
        train_best_classwise_recall_image_level = []
        train_best_classwise_f1_image_level = []
        train_best_acc = 0.0
        train_best_f1 = 0.0
        train_best_classwise_precision = []
        train_best_classwise_recall = []
        train_best_classwise_f1 = []
        lowest_val_loss = 12345.0
        lowest_val_loss_epoch = 0
        for epoch in range(1,self.epochs + 1):
            if (True == self.LE):
                for p in self.model.model.parameters():
                    p.requires_grad = False
            #train epoch
            weighted_f1_image_level, accuracy_image_level,classwise_precision_image_level,classwise_recall_image_level,classwise_f1_image_level, weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1, loss = self.train_epoch_bach()
            #evaluate on validation set
            val_weighted_f1_image_level, val_accuracy_image_level, val_classwise_precision_image_level,val_classwise_recall_image_level,val_classwise_f1_image_level, val_weighted_f1, val_accuracy, val_classwise_precision,val_classwise_recall,val_classwise_f1, val_loss = self.evaluate_validation_set_bach()
            print("Epoch {}/{} Train Loss:{}, Val Loss: {}".format(epoch, self.epochs, loss, val_loss))
            if (epoch - lowest_val_loss_epoch) >= self.early_stopping_patience:
                print('Eearly stopping criteria matched - training stop at highest validation Accuracy, validation loss, and epoch', best_acc, val_loss, epoch)
                break
            if lowest_val_loss > val_loss:
                lowest_val_loss = val_loss
                lowest_val_loss_epoch = epoch        
            if best_acc < val_accuracy:
                best_acc_image_level = val_accuracy_image_level
                best_f1_image_level = val_weighted_f1_image_level
                best_classwise_precision_image_level = val_classwise_precision_image_level
                best_classwise_recall_image_level = val_classwise_recall_image_level
                best_classwise_f1_image_level = val_classwise_f1_image_level
                best_acc = val_accuracy
                best_f1 = val_weighted_f1
                best_classwise_precision = val_classwise_precision
                best_classwise_recall = val_classwise_recall
                best_classwise_f1 = val_classwise_f1
                #train
                train_best_acc_image_level = accuracy_image_level
                train_best_f1_image_level = weighted_f1_image_level
                train_best_classwise_precision_image_level = classwise_precision_image_level
                train_best_classwise_recall_image_level = classwise_recall_image_level
                train_best_classwise_f1_image_level = classwise_f1_image_level
                train_best_acc = accuracy
                train_best_f1 = weighted_f1
                train_best_classwise_precision = classwise_precision
                train_best_classwise_recall = classwise_recall
                train_best_classwise_f1 = classwise_f1
                result_path = f"{self.result_folder}/{self.experiment_description}"
                Path(result_path).mkdir(parents=True, exist_ok=True)
                for file in Path(result_path).glob('*'):
                    file.unlink()
                torch.save(self.model.state_dict(), f"{result_path}/_{epoch}_{val_accuracy_image_level}_{val_accuracy}_{val_weighted_f1}.pth")
                best_model_weights_path = f"{result_path}/_{epoch}_{val_accuracy_image_level}_{val_accuracy}_{val_weighted_f1}.pth"
            if None != self.scheduler:
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
            if 4 == self.num_classes:
                for index in range(0,len(bc_config.bach_label_list)):
                    self.writer.add_scalar(f'F1/Validation_Set/{bc_config.bach_label_list[index]}', val_classwise_f1[index], epoch)
                    self.writer.add_scalar(f'F1/Training_Set/{bc_config.bach_label_list[index]}', classwise_f1[index], epoch)
                    self.writer.add_scalar(f'Precision/Validation_Set/{bc_config.bach_label_list[index]}', val_classwise_precision[index], epoch)
                    self.writer.add_scalar(f'Precision/Training_Set/{bc_config.bach_label_list[index]}', classwise_precision[index], epoch)
                    self.writer.add_scalar(f'Recall/Validation_Set/{bc_config.bach_label_list[index]}', val_classwise_recall[index], epoch)
                    self.writer.add_scalar(f'Recall/Training_Set/{bc_config.bach_label_list[index]}', classwise_recall[index], epoch)
        #test
        test_weighted_f1_image_level, test_accuracy_image_level, test_classwise_precision_image_level, test_classwise_recall_image_level, test_classwise_f1_image_level, test_weighted_f1, test_accuracy, test_classwise_precision, test_classwise_recall, test_classwise_f1 = self.evaluate_testset_bach(model_weights_path = best_model_weights_path)
        result_dict = {
            "train" : [train_best_acc_image_level, train_best_f1_image_level, train_best_classwise_precision_image_level, train_best_classwise_recall_image_level, train_best_classwise_f1_image_level, train_best_acc, train_best_f1, train_best_classwise_precision, train_best_classwise_recall, train_best_classwise_f1],
            "val" : [best_acc_image_level, best_f1_image_level, best_classwise_precision_image_level, best_classwise_recall_image_level, best_classwise_f1_image_level, best_acc, best_f1, best_classwise_precision, best_classwise_recall, best_classwise_f1],
            "test" : [test_accuracy_image_level, test_weighted_f1_image_level, test_classwise_precision_image_level, test_classwise_recall_image_level, test_classwise_f1_image_level, test_weighted_f1, test_accuracy, test_classwise_precision, test_classwise_recall, test_classwise_f1]
        }
        return result_dict
    
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

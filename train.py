
import torch
from torch import nn
from torch.utils.data import DataLoader ,Dataset
import matplotlib.pyplot as plt
from os.path import join
import pandas as pd
from dataclasses import dataclass
import cv2
import albumentations as A
import numpy as np
from rgb_segmentation import get_corresponding_Color
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F

# from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class SemSegDataset(Dataset):

    def __init__(self ,data_path ,images_folder ,masks_folder ,csv_path ,dataset_type ,num_classes ,validset_ratio
                 ,transform=None ,class_names=None):
        self.data_path = data_path
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.csv_path = csv_path
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.transform = transform
        self.class_names = class_names


        self.image_ids = pd.read_csv(join(data_path ,csv_path)).astype('str')
        if dataset_type == 'train' or dataset_type == 'valid':

            train_set = self.image_ids.sample(frac= 1 -validset_ratio)
            valid_set = self.image_ids.drop(train_set.index)



            if dataset_type == 'train':
                self.dataset = train_set
            else:
                self.dataset = valid_set

        elif dataset_type == 'test':
            self.dataset = self.image_ids
        else:
            raise Exception("Wrong dataset type")

        self.dataset.reset_index(inplace=True ,drop=True)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self ,index):
        img = cv2.cvtColor \
            (cv2.imread(f"{join(self.images_folder ,self.dataset.iloc[index]['ImageID'])}.jpg" ,cv2.IMREAD_UNCHANGED)
            ,cv2.COLOR_BGR2RGB)

        if self.dataset_type != 'test':
            mask = cv2.imread(f"{join(self.masks_folder ,self.dataset.iloc[index]['ImageID'])}.png"
                              ,cv2.IMREAD_UNCHANGED)

            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                return transformed['image'] ,transformed['mask']
            else:
                return img ,mask
        else:
            if self.transform is not None:
                return self.transform(image=img)
            else:
                return img


@dataclass
class DatasetInfo:
    data_path :str
    images_folder :str
    masks_folder :str
    train_file :str
    test_file :str
    num_classes :int
    validset_ratio :float
    mean :list[float]
    std :list[float]
    class_names :list[str]


@dataclass
class TrainingConfiguration:
    epochs :int
    batch_size :int
    accumulate_every :int
    lr :float
    weight_decay :float
    pretrained :bool
    optimizer :str
    criterion :str
    transform :A.Compose
    experiment_name :str
    class_weights :list[float]


if __name__ == '__main__':

    datasetInfo = DatasetInfo(
        data_path=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2',
        images_folder=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\imgs\imgs',
        masks_folder=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\masks\masks',
        train_file=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\train.csv',
        test_file=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\test.csv',
        num_classes=12,
        validset_ratio=0.20,
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
        class_names=['Background' ,'Person' ,'Bike' ,'Car' ,'Drone' ,'Boat' ,'Animal' ,'Obstacle' ,'Construction'
                     ,'Vegetation' ,'Road' ,'Sky']

    )

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.1),
        A.ElasticTransform(p=0.1),
        A.Perspective(p=0.1),
        A.PixelDropout(0.009 ,p=0.1),
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0,
                           shift_limit=0.2, p=0.5, border_mode=0),
        A.Normalize(datasetInfo.mean, datasetInfo.std),
        A.ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(datasetInfo.mean, datasetInfo.std),
        A.ToTensorV2()
    ])

    trainingConfiguration = TrainingConfiguration(
        epochs=40,
        batch_size=4,
        accumulate_every= 4,
        lr=1e-4,
        weight_decay=0,
        pretrained=True,
        optimizer="Adam",
        criterion="crossEntropy with dice and aux loss class weights [0.6321080603794254, 2.6862471966606334, 4.822572602267548, 3.9259031015315164, 6.3766972286223, 6.996623402678927, 5.840580600554346, 2.884140684001012, 1.938011612195458, 0.5, 0.5496670838750093, 2.1350857456239694]",
        transform=train_transform,
        experiment_name="exp14",
        class_weights = torch.tensor([1.0, 3.0, 5.0, 4.0, 6, 7, 6, 3, 2, 1, 1, 2] ,device='cuda')
    )


    train_set = SemSegDataset(
        data_path=datasetInfo.data_path,
        images_folder=datasetInfo.images_folder,
        masks_folder=datasetInfo.masks_folder,
        csv_path=datasetInfo.train_file,
        dataset_type='train',
        num_classes=datasetInfo.num_classes,
        validset_ratio=datasetInfo.validset_ratio,
        transform=train_transform,
        class_names=datasetInfo.class_names
    )

    valid_set = SemSegDataset(
        data_path=datasetInfo.data_path,
        images_folder=datasetInfo.images_folder,
        masks_folder=datasetInfo.masks_folder,
        csv_path=datasetInfo.train_file,
        dataset_type='valid',
        num_classes=datasetInfo.num_classes,
        validset_ratio=datasetInfo.validset_ratio,
        transform=test_transform,
        class_names=datasetInfo.class_names
    )

    # test_set = SemSegDataset(
    #     data_path=datasetInfo.data_path,
    #     images_folder=datasetInfo.images_folder,
    #     masks_folder=datasetInfo.masks_folder,
    #     csv_path=datasetInfo.test_file,
    #     dataset_type='test',
    #     num_classes=datasetInfo.num_classes,
    #     validset_ratio=datasetInfo.validset_ratio,
    #     transform=transform,
    #     class_names=datasetInfo.class_names
    # )









    # cpu_count=16
    train_loader = DataLoader(train_set ,shuffle=True ,num_workers=12 ,persistent_workers=True
                              ,batch_size=trainingConfiguration.batch_size ,drop_last=True)
    valid_loader = DataLoader(valid_set ,shuffle=False ,num_workers=3 ,persistent_workers=False
                              ,batch_size=trainingConfiguration.batch_size ,drop_last=True)


    def dice_coef_loss(predictions, ground_truths, num_classes=2, dims=(1, 2), smooth=1e-8):
        """
        Calculate a combined loss function comprising the Naive Dice coefficient loss and cross-entropy loss.

        Arguments:
        predictions (torch.tensor): Prediction (P) model output logits.
                                    Shape: [batch_size(B), num_classes, height(H), width(W)]

        ground_truths (torch.tensor): Ground truth mask (G). [B, num_classes, H, W].

        dims (tuple): Dimensions corresponding to image height and width in a tensor shape: [B, H, W, num_classes].

        Returns:
        torch.tensor: A scalar tensor representing the combined Naive Mean Dice coefficient and cross-entropy loss.
        """

        # Convert single channel ground truth masks into one-hot encoded tensor.
        # Shape: (B, H, W, num_classes)
        ground_truth_oh = torch.nn.functional.one_hot(ground_truths, num_classes=num_classes)

        # Normalize model predictions and transpose.
        # Shape :: [B, num_classes, H, W] --> [B, H, W, num_classes]
        # This is done to match the shape of ground_truth_oh.
        prediction_norm = torch.nn.functional.softmax(predictions, dim=1).permute(0, 2, 3, 1)

        # Intersection: |G ∩ P|. Shape: [B, num_classes]
        intersection = (prediction_norm * ground_truth_oh).sum(dim=dims)

        # Summation: |G| + |P|. Shape: [B, num_classes].
        summation = (prediction_norm.sum(dim=dims) + ground_truth_oh.sum(dim=dims))

        # Dice Shape: [B, num_classes]
        dice = (2.0 * intersection + smooth) / (summation + smooth)

        # Compute the mean over the remaining axes (batch and classes).
        dice_mean = dice.mean()

        # Compute cross-entropy loss.
        CE = torch.nn.functional.cross_entropy(predictions, ground_truths ,weight=trainingConfiguration.class_weights)

        return (1.0 - dice_mean ) *0.8 + CE, dice_mean.detach()


    def mean_iou(predictions, ground_truths, num_classes=2, dims=(1, 2)):
        predictions = predictions.argmax(dim=1)
        """
        Arguments:
        predictions (torch.tensor): Prediction (P) from the model with or without softmax.
                                    Shape: [batch_size(B), height(H), width(W)]

        ground_truths (torch.tensor): Ground truth mask (G). Shape: [B, H, W]

        dims (tuple): Dimensions corresponding to image height and width in a tensor shape: [B, H, W, C].

        Returns:
        torch.tensor: A scalar tensor representing the Classwise Mean IoU metric.
        """

        # Convert single channel ground truth masks into one-hot encoded tensor.
        # Shape: [B, H, W] --> [B, H, W, num_classes]
        ground_truths = F.one_hot(ground_truths, num_classes=num_classes)

        # Converting unnormalized predictions into one-hot encoded across channels.
        # Shape: [B, H, W] --> [B, H, W, num_classes]
        predictions = F.one_hot(predictions, num_classes=num_classes)

        # Intersection: |G ∩ P|. Shape: [B, num_classes]
        intersection = (predictions * ground_truths).sum(dim=dims)

        # Summation: |G| + |P|. Shape: [B, num_classes].
        summation = (predictions.sum(dim=dims) + ground_truths.sum(dim=dims))

        # Union. Shape: [B, num_classes]
        union = summation - intersection

        # IoU Shape: [B, num_classes]
        iou = intersection / union

        # As no smoothing is used we replace any 'nan' value that with 0.
        # With smoothing the results yields slightly different values.
        iou = torch.nan_to_num(iou, nan=0.0)

        # Shape: [batch_size,]
        num_classes_present = torch.count_nonzero(summation, dim=1)
        # IoU per image.
        # Average over the total number of classes present in ground_truths and predictions.
        # Shape: [batch_size,]
        iou = iou.sum(dim=1) / num_classes_present

        # Compute the mean over the remaining axes (batch and classes).
        # Shape: Scalar
        iou_mean = iou.mean()

        return iou_mean

    def params_status(model ,freeze=False):
        for param in model.parameters():
            param.requires_grad = not freeze


    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT ,aux_loss=True)

    model.classifier[4] = nn.Conv2d(256, datasetInfo.num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, datasetInfo.num_classes, kernel_size=(1, 1), stride=(1, 1))

    params_status(model,True)
    params_status(model.classifier,False)
    params_status(model.aux_classifier,False)
    params_status(model.backbone['layer4'],False)



    print(summary(model))






    # model.aux_classifier[4] = nn.Conv2d(10, datasetInfo.num_classes, kernel_size=(1, 1), stride=(1, 1))
    # params_status(model.classifier,False)











    writer = SummaryWriter(f'./runs/{trainingConfiguration.experiment_name}')
    writer.add_text("training Configuration" ,f"{trainingConfiguration}")

    optimizer = torch.optim.Adam(model.parameters() ,lr=trainingConfiguration.lr
                                 ,weight_decay=trainingConfiguration.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.9)


    def initialize_grad_stats():
        # Initialize containers for tracking gradient statistics
        return {'means': {}, 'maxs': {}, 'counts': {} ,'mins' :{}}


    def update_grad_stats(model, stats):
        # Update running statistics for gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_abs = param.grad.abs()

                # Initialize trackers for this parameter if not already present
                if name not in stats['means']:
                    stats['means'][name] = 0.0
                    stats['maxs'][name] = 0.0
                    stats['counts'][name] = 0
                    stats['mins'][name] = 0

                # Update running statistics
                stats['means'][name] += grad_abs.mean().item()
                stats['maxs'][name] = max(stats['maxs'][name], grad_abs.max().item())
                stats['mins'][name] = min(stats['mins'][name], grad_abs.min().item())
                stats['counts'][name] += 1


    def log_grad_stats(stats, tb_writer, epoch):
        # Log the average statistics to TensorBoard
        for name in stats['means'].keys():
            if stats['counts'][name] > 0:
                avg_mean = stats['means'][name] / stats['counts'][name]
                tb_writer.add_scalar(f'grad_stats/{name}_mean', avg_mean, epoch)
                tb_writer.add_scalar(f'grad_stats/{name}_max', stats['maxs'][name], epoch)
                tb_writer.add_scalar(f'grad_stats/{name}_min', stats['mins'][name], epoch)


    def train(trainingConfiguration,
              class_names,
              train_loader,
              valid_loader,
              model,
              optimizer :torch.optim,
              tb_writer :SummaryWriter,
              num_classes=12):

        model = model.to('cuda')
        previous_loss = 0
        best_loss = float('inf')

        for e in range(trainingConfiguration.epochs + 1):
            model = model.train()
            training_loss = 0
            running_iou = torch.zeros(num_classes ,device='cuda')
            dc = 0
            m_iou = 0
            step = 0
            grad_stats = initialize_grad_stats()
            for images, masks in (tqdm(train_loader, desc='Training:', total=len(train_loader))):
                step += 1
                images = images.to('cuda')
                masks = masks.to('cuda').long()

                with torch.amp.autocast('cuda'):
                    output = model(images)
                    main_loss ,main_mean_dice = dice_coef_loss(output['out'] ,masks ,12)
                    aux_loss ,_ = dice_coef_loss(output['aux'] ,masks ,12)
                    loss = main_loss + 0.2 * aux_loss

                loss.backward()

                if step % trainingConfiguration.accumulate_every == 0 or step == len(train_loader):
                    optimizer.step()
                    if e % 5 == 0:
                        update_grad_stats(model, grad_stats)

                    optimizer.zero_grad()


                m_iou += mean_iou(output['out'].long() ,masks.long() ,12)
                training_loss += loss.item()
                dc += main_mean_dice



            training_loss /= len(train_loader)
            dc /= len(train_loader)
            m_iou /= len(train_loader)
            lr_scheduler.step(training_loss)

            if 0 < previous_loss - training_loss <= 0.03 and optimizer.param_groups[0]['lr'] <= 0.1:
                optimizer.param_groups[0]['lr'] *= 1.1

            previous_loss = training_loss


            tb_writer.add_scalar('Loss/train' ,training_loss ,e)
            tb_writer.add_scalars('iou_per_class/training'
                                  ,{class_names[i] :running_iou[i].item() for i in range(num_classes)} ,e)
            tb_writer.add_scalar('Metrics/mean_iou/training' ,m_iou ,e)
            tb_writer.add_scalar('Metrics/mean_dice_coefficient/training' ,dc ,e)
            tb_writer.add_scalar('Metrics/lr_over_epochs' ,optimizer.param_groups[0]['lr'] ,e)

            if e % 5 == 0:
                valid_loss = validate(class_names ,valid_loader ,model ,tb_writer ,e)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(model.state_dict(), f'{trainingConfiguration.experiment_name}_best.pth')

                log_grad_stats(grad_stats, tb_writer, e)


    def validate(class_names ,valid_loader ,model ,tb_writer ,e ,num_classes=12):
        model = model.to('cuda')
        model = model.eval()
        running_loss = 0
        running_iou = torch.zeros(num_classes ,device='cuda')
        dc = 0
        m_iou =0
        with torch.no_grad():

            for images ,masks in tqdm(valid_loader ,desc="Validation: " ,total=len(valid_loader)):
                images = images.to('cuda')
                masks = masks.to('cuda').long()
                output = model(images)
                main_loss, main_mean_dice = dice_coef_loss(output['out'], masks, 12)
                aux_loss, _ = dice_coef_loss(output['aux'], masks, 12)
                loss = main_loss + 0.2 * aux_loss


                m_iou += mean_iou(output['out'].long() ,masks.long() ,12)
                running_loss += loss.item()
                dc += main_mean_dice

            running_loss /= len(valid_loader)
            running_iou  /= len(valid_loader)
            m_iou /= len(valid_loader)
            dc /= len(valid_loader)
            tb_writer.add_scalar('Loss/validation' ,running_loss ,e)
            tb_writer.add_scalar('Metrics/mean_iou/validation' ,m_iou ,e)
            tb_writer.add_scalar('Metrics/mean_dice_coefficient/validation' ,dc ,e)
        return running_loss





    train(trainingConfiguration ,datasetInfo.class_names ,train_loader ,valid_loader ,model ,optimizer ,writer)

    torch.save(model.state_dict(), f'{trainingConfiguration.experiment_name}.pth')
    torch.save(optimizer.state_dict(), f'{trainingConfiguration.experiment_name}_optimizer.pth')














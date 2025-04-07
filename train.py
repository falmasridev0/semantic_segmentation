import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
import segmentation_models_pytorch as smp


# from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class SemSegDataset(Dataset):

    def __init__(self, data_path, images_folder, masks_folder, csv_path, dataset_type, num_classes, validset_ratio
                 , transform=None, class_names=None):
        self.data_path = data_path
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.csv_path = csv_path
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.transform = transform
        self.class_names = class_names

        self.image_ids = pd.read_csv(join(data_path, csv_path)).astype('str')
        if dataset_type == 'train' or dataset_type == 'valid':

            train_set = self.image_ids.sample(frac=1 - validset_ratio)
            valid_set = self.image_ids.drop(train_set.index)

            if dataset_type == 'train':
                self.dataset = train_set
            else:
                self.dataset = valid_set

        elif dataset_type == 'test':
            self.dataset = self.image_ids
        else:
            raise Exception("Wrong dataset type")

        self.dataset.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = cv2.cvtColor \
            (cv2.imread(f"{join(self.images_folder, self.dataset.iloc[index]['ImageID'])}.jpg", cv2.IMREAD_UNCHANGED)
             , cv2.COLOR_BGR2RGB)

        if self.dataset_type != 'test':
            mask = cv2.imread(f"{join(self.masks_folder, self.dataset.iloc[index]['ImageID'])}.png"
                              , cv2.IMREAD_UNCHANGED)

            if self.transform is not None:
                transformed = self.transform(image=img, mask=mask)
                return transformed['image'], transformed['mask']
            else:
                return img, mask
        else:
            if self.transform is not None:
                return self.transform(image=img)
            else:
                return img


@dataclass
class DatasetInfo:
    data_path: str
    images_folder: str
    masks_folder: str
    train_file: str
    test_file: str
    num_classes: int
    validset_ratio: float
    mean: list[float]
    std: list[float]
    class_names: list[str]


@dataclass
class TrainingConfiguration:
    epochs: int
    batch_size: int
    accumulate_every: int
    lr: float
    weight_decay: float
    pretrained: bool
    optimizer: str
    criterion: str
    transform: A.Compose
    experiment_name: str
    class_weights: list[float]


if __name__ == '__main__':

    datasetInfo = DatasetInfo(
        data_path=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2',
        images_folder=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\imgs\imgs',
        masks_folder=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\masks\masks',
        train_file=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\train.csv',
        test_file=r'C:\Users\falmasridev\Documents\opencv_courses\c2\opencv-pytorch-segmentation-project-round2\test.csv',
        num_classes=12,
        validset_ratio=0.2,
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
        class_names=['Background', 'Person', 'Bike', 'Car', 'Drone', 'Boat', 'Animal', 'Obstacle', 'Construction'
            , 'Vegetation', 'Road', 'Sky']

    )

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ElasticTransform(p=0.2),
        A.Perspective(p=0.3),
        A.PixelDropout(0.009, p=0.3),
        A.Affine(
            scale=(0.8, 1.2),  # Scale between 80% and 120%
            translate_percent=(0.05, 0.05),  # Shift up to 10% in both x and y
            rotate=(-6, 6),  # Rotate between -15° and 15°
            shear=(-5, 5),  # Shear between -10° and 10°
            p=0.4 ),
        A.Normalize(datasetInfo.mean, datasetInfo.std),
        A.ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(datasetInfo.mean, datasetInfo.std),
        A.ToTensorV2()
    ])

    trainingConfiguration = TrainingConfiguration(
        epochs=60,
        batch_size=4,
        accumulate_every=4,
        lr=1e-4,
        weight_decay=0.0007,
        pretrained=True,
        optimizer="Adam",
        criterion="crossEntropy with dice and aux loss class weights [0.6321080603794254, 2.6862471966606334, 4.822572602267548, 3.9259031015315164, 6.3766972286223, 6.996623402678927, 5.840580600554346, 2.884140684001012, 1.938011612195458, 0.5, 0.5496670838750093, 2.1350857456239694]",
        transform=train_transform,
        experiment_name="exp20",
        class_weights=torch.tensor([1, 2.6862471966606334, 4.822572602267548, 3.9259031015315164, 6.3766972286223, 6.996623402678927, 5.840580600554346, 2.884140684001012, 1.938011612195458, 1, 1, 2.1350857456239694], device='cuda')
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



    # cpu_count=16
    train_loader = DataLoader(train_set, shuffle=True, num_workers=12, persistent_workers=True
                              , batch_size=trainingConfiguration.batch_size, drop_last=True)
    valid_loader = DataLoader(valid_set, shuffle=False, num_workers=3, persistent_workers=False
                              , batch_size=trainingConfiguration.batch_size, drop_last=True)




    def params_status(model, freeze=False):
        for param in model.parameters():
            param.requires_grad = not freeze


    model = smp.Unet(classes=datasetInfo.num_classes,encoder_weights="imagenet")

    params_status(model, False)

    print(summary(model))



    writer = SummaryWriter(f'./runs/{trainingConfiguration.experiment_name}')
    writer.add_text("training Configuration", f"{trainingConfiguration}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=trainingConfiguration.lr
                                 , weight_decay=trainingConfiguration.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.85)


    def initialize_grad_stats():
        # Initialize containers for tracking gradient statistics
        return {'means': {}, 'maxs': {}, 'counts': {}, 'mins': {}}


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


    dice_loss = smp.losses.DiceLoss(smp.losses.constants.MULTICLASS_MODE, None, False, True, 0, None)
    ce = torch.nn.CrossEntropyLoss(weight=trainingConfiguration.class_weights)
    focal_loss = smp.losses.FocalLoss(smp.losses.constants.MULTICLASS_MODE, alpha=0.2)

    def train(trainingConfiguration,
              class_names,
              train_loader,
              valid_loader,
              model,
              optimizer: torch.optim,
              tb_writer: SummaryWriter,
              num_classes=12):

        model = model.to('cuda')
        previous_loss = 0
        best_loss = float('inf')
        scaler = torch.amp.GradScaler('cuda')
        for e in range(trainingConfiguration.epochs + 1):
            model = model.train()
            training_loss = 0
            # running_dice_loss = 0
            # running_ce_loss = 0
            tp, fp, fn, tn = (torch.zeros((trainingConfiguration.batch_size,datasetInfo.num_classes), device='cuda') for _ in range(4))

            step = 0
            grad_stats = initialize_grad_stats()
            for images, masks in (tqdm(train_loader, desc='Training:', total=len(train_loader))):
                step += 1
                images = images.to('cuda').float()
                masks = masks.to('cuda').long()

                with torch.amp.autocast('cuda'):
                    # output, _ = model(images)
                    # main_loss = dice_loss(output, masks)
                    # ce_loss = ce.forward(output, masks)
                    # loss = main_loss + ce_loss
                    output = model(images)

                    loss = focal_loss(output,masks)
                    batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(torch.argmax(output.to('cuda'),dim=1), masks,
                                                                                   smp.losses.constants.MULTICLASS_MODE,num_classes=datasetInfo.num_classes)
                    tp += batch_tp.to('cuda')
                    fp += batch_fp.to('cuda')
                    fn += batch_fn.to('cuda')
                    tn += batch_tn.to('cuda')

                scaler.scale(loss).backward()

                if step % trainingConfiguration.accumulate_every == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()

                    if e % 5 == 0:
                        update_grad_stats(model, grad_stats)

                    optimizer.zero_grad()

                training_loss += loss.item()
                # running_dice_loss += main_loss.item()
                # running_ce_loss += ce_loss.item()

            training_loss /= len(train_loader)
            # running_dice_loss /= len(train_loader)
            # running_ce_loss /= len(train_loader)

            lr_scheduler.step(training_loss)

            if 0 < previous_loss - training_loss <= 0.03 and optimizer.param_groups[0]['lr'] <= 0.001 :
                optimizer.param_groups[0]['lr'] *= 1.2

            previous_loss = training_loss

            tb_writer.add_scalar('Loss/train', training_loss, e)
            # tb_writer.add_scalar('Loss/dice_loss/training', running_dice_loss, e)
            # tb_writer.add_scalar('Loss/CrossEntropyLoss/training', running_ce_loss, e)
            tb_writer.add_scalar('Metrics/mean_iou/training', smp.metrics.iou_score(tp, fp, fn, tn,reduction='micro-imagewise').item(), e)
            tb_writer.add_scalar('Metrics/f1_score/training', smp.metrics.f1_score(tp, fp, fn, tn,reduction='micro-imagewise').item(), e)
            tb_writer.add_scalars('iou_per_class/training'
                                  ,{class_names[i] : smp.metrics.iou_score(tp, fp, fn, tn).mean(dim=0)[i].item() for i in range(num_classes)} ,e)
            tb_writer.add_scalar('Metrics/lr_over_epochs', optimizer.param_groups[0]['lr'], e)

            if e % 5 == 0:
                valid_loss = validate(class_names, valid_loader, model, tb_writer, e)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    model.save_pretrained(f'./{trainingConfiguration.experiment_name}_best')

                log_grad_stats(grad_stats, tb_writer, e)


    def validate(class_names, valid_loader, model, tb_writer, e, num_classes=12):
        model = model.to('cuda')
        model = model.eval()
        running_loss = 0
        # running_dice_loss = 0
        # running_ce_loss = 0
        tp, fp, fn, tn = (torch.zeros((trainingConfiguration.batch_size, datasetInfo.num_classes), device='cuda') for _
                          in range(4))

        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc="Validation: ", total=len(valid_loader)):
                images = images.to('cuda').float()
                masks = masks.to('cuda').long()
                # output, _ = model(images)
                # main_loss = dice_loss(output, masks)
                # ce_loss = ce.forward(output, masks)
                # loss = main_loss + ce_loss
                output = model(images)
                loss = focal_loss(output,masks)

                batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(torch.argmax(output.to('cuda'), dim=1),
                                                                               masks,
                                                                               smp.losses.constants.MULTICLASS_MODE,
                                                                               num_classes=datasetInfo.num_classes)
                tp += batch_tp.to('cuda')
                fp += batch_fp.to('cuda')
                fn += batch_fn.to('cuda')
                tn += batch_tn.to('cuda')

                running_loss += loss.item()
                # running_dice_loss += main_loss.item()
                # running_ce_loss += ce_loss.item()

            running_loss /= len(valid_loader)
            # running_dice_loss /= len(valid_loader)
            # running_ce_loss /= len(valid_loader)

            tb_writer.add_scalar('Loss/validation', running_loss, e)
            # tb_writer.add_scalar('Loss/dice_loss/validation', running_dice_loss, e)
            # tb_writer.add_scalar('Loss/CrossEntropyLoss/validation', running_ce_loss, e)
            tb_writer.add_scalar('Metrics/mean_iou/validation', smp.metrics.iou_score(tp, fp, fn, tn,reduction='micro-imagewise').item(), e)
            tb_writer.add_scalar('Metrics/f1_score/validation', smp.metrics.f1_score(tp, fp, fn, tn,reduction='micro-imagewise').item(), e)
            tb_writer.add_scalars('iou_per_class/validation'
                                  ,{class_names[i] : smp.metrics.iou_score(tp, fp, fn, tn).mean(dim=0)[i].item() for i in range(num_classes)} ,e)
        return running_loss


    train(trainingConfiguration, datasetInfo.class_names, train_loader, valid_loader, model, optimizer, writer)

    model.save_pretrained(f'./{trainingConfiguration.experiment_name}_last')

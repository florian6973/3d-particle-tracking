from __future__ import print_function, division
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')


import shutil

from dl_conf import *
import os

def build_folder(dataset, dataset_data, folderset):
    #for i in range(0, nclasses): # all classes
    #    folder = os.path.join(folderset, str(i))
    #    if not os.path.exists(folder):
    #        os.mkdir(folder)
    # Found no valid file for the classes 62, 64, 65. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

    for type, n, v, i, f, img in dataset:
        dest = get_filepath(type, n, v, i, f)
        filename = dest
        data_annot = dataset_data[(dataset_data['type'] == type)
                                  & (dataset_data['n'] == n)
                                  & (dataset_data['v'] == v)
                                  & (dataset_data['i'] == i)
                                  & (dataset_data['f'] == f)][['x', 'y', 'r', 'top']]
        #print(data_annot)
        nbtop = data_annot[data_annot['top']]['top'].count()
        #print(nbtop)
        folder = os.path.join(folderset, str(nbtop))
        if not os.path.exists(folder):
            os.mkdir(folder)
        output_file = os.path.join(folder, os.path.basename(filename))
        print("Copying {} to {}".format(filename, output_file))
        shutil.copy(filename, output_file)

def build_dataset(datadir):
    datadirtrain = os.path.join(datadir, 'train')
    datadirval = os.path.join(datadir, 'val')
    if not os.path.exists(datadir):
        os.mkdir(datadir)
        os.mkdir(datadirtrain)
        os.mkdir(datadirval)

    dataset_data = get_data_all_beads()
    train, test = split_test_train2()

    print('Building train dataset')
    build_folder(train, dataset_data, datadirtrain)
    print('Building test dataset')
    build_folder(test, dataset_data, datadirval)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from torchvision.models import ResNet18_Weights

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


cudnn.benchmark = True
plt.ion()   # interactive mode

def load_data(data_dir):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(),
            transforms.RandomAutocontrast()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(),
            transforms.RandomAutocontrast()
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config['batch_size'],
                                                  shuffle=True, num_workers=4)#1)#4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = 'Ntop'#image_datasets['train'].classes
    print(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    return dataloaders, dataset_sizes, class_names, device

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    #best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs, 1)
                    #print(outputs.shape)
                    #_, preds = torch.max(outputs, 1)
                    #print(outputs, labels)
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}') #Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss: #epoch_acc > best_acc:
                #best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best loss Acc: {best_loss:4f}')
    #print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloaders, num_images=15):
    try:
        was_training = model.training
    except:
        was_training = False
    model.eval()
    images_so_far = 0
    #fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                plt.figure()
                #ax = plt.subplot(num_images//2, 2, images_so_far)
                #ax.axis('off')
                #ax.set_title(f'predicted: {class_names[preds[j]]}, real: {class_names[labels[j]]}')
                plt.title(f'predicted: {outputs[j][0]}, real: {labels[j]}')
                imshow(inputs.cpu().data[j])
                plt.savefig(f'imgs/{images_so_far}.png')

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=nn.pyining)

def append_dropout(model, rate=0.2):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=True))
            setattr(model, name, new)

def build_model_old():
    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    #for param in model_ft.parameters():
    #    param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 1)
    #append_dropout(model_ft)

    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])#1e-2) #optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config['step_size'], gamma=config['gamma'])
    #scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.99)#gamma=0.9)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler

def create_model():
    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    #for param in model_ft.parameters():
    #    param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 1)
    #append_dropout(model_ft)

    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)
    return model_ft
    pass

def launch():
    mp = 'last_model.pt'
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = build_model_old()
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=24)#55)
    torch.save(model_ft.state_dict(), mp)
    model_ft = create_model()
    model_state = torch.load(mp)
    model_ft.load_state_dict(model_state)
    #model_ft.eval()
    visualize_model(model_ft, dataloaders)

def test_accuracy(net, device="cpu"):
    trainset, testset = load_data2(data_dir)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0.
    total = 0.
    maxerr = 0.
    lbl = 0.
    rem = 0.
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            outputs = torch.squeeze(outputs, 1)
            print(outputs, labels.float())
            se = (outputs - labels.float()).abs()
            lm = se.max().item()
            if lm > maxerr:
                maxerr = lm
                lbl = (labels[se.argmax()].item(), outputs[se.argmax()].item())
            # compute relative error
            if labels.min().item() > 20:
                err = (outputs - labels.float())/labels.float()
                rem = max(rem, err.abs().max().item())
                print("re", err)
            print(se)
            correct += torch.sum(se**2).item()
            total += se.size(0)
            #correct += (predicted == labels).sum().item()

    rmse = np.sqrt(correct / total)
    print(rmse)
    print(lbl, maxerr)
    print(rem)
    return rmse

def build_model(config):
    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    #for param in model_ft.parameters():
    #    param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    
    torch.manual_seed(3)
    model_ft.fc = nn.Linear(num_ftrs, 1)
    #append_dropout(model_ft)

    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])#1e-2) #optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config['step_size'], gamma=config['gamma'])
    #scheduler = lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.99)#gamma=0.9)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler



def load_data2(data_dir):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(),
            transforms.RandomAutocontrast()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomVerticalFlip(),
            transforms.RandomAutocontrast()
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    return image_datasets['train'], image_datasets['val']

def train_cifar(config, checkpoint_dir=None, data_dir=None):
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = build_model(config)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_ft.load_state_dict(model_state)
        optimizer_ft.load_state_dict(optimizer_state)

    trainset, testset = load_data2(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times # TODO hyperparams?
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            # forward + backward + optimize
            outputs = model_ft(inputs)
            outputs = torch.squeeze(outputs, 1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer_ft.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model_ft(inputs)
                outputs = torch.squeeze(outputs, 1)
                #_, predicted = torch.max(outputs.data, 1)
                #total += labels.size(0)
                #correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels.float())
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model_ft.state_dict(), optimizer_ft.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=4):
    #data_dir = r'/n/holyscratch01/bertoldi_lab/fpollet/raw_data/pt_data'
    load_data2(data_dir)
    config = {
        'step_size': tune.choice([1, 5, 10, 50]),
        'gamma': tune.choice([0.1, 0.5, 0.9, 0.99]),
        "weight_decay": tune.loguniform(1e-5, 1e1),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 0, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))

def test():
    mp = r'/n/home08/fpollet/ray_results/train_cifar_2022-06-25_14-51-55/train_cifar_e2308_00000_0_batch_size=2,gamma=0.1000,lr=0.0002,step_size=50,weight_decay=0.1392_2022-06-25_14-51-55/checkpoint_000009/checkpoint'
    
    dataloaders, dataset_sizes, class_names, device = load_data(data_dir)
    
    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    #for param in model_ft.parameters():
    #    param.requires_grad = False

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    
    torch.manual_seed(3)
    model_ft.fc = nn.Linear(num_ftrs, 1)
    #append_dropout(model_ft)

    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.to(device)

    model_state,_ = torch.load(mp)
    model_ft.load_state_dict(model_state)

    #visualize_model(model_ft, dataloaders)
    test_accuracy(model_ft, device)


if __name__ == "__main__":
    #datadir = r'H:\Cluster\pt_data'
    data_dir = r'/n/holyscratch01/bertoldi_lab/fpollet/raw_data/pt_data'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #main()
    #visualize_model
    #retrain and save
    #test()

    #nclasses = 66 # to fix, val dataset?
    #build_dataset(datadir)
    
    config={
        'step_size': 12, #15
        'gamma': 0.1,
        "weight_decay": 1, # 10 block around 300
        "lr": 0.0002,
        "batch_size": 16
    } # should decreqse generalisation? weird?

    dataloaders, dataset_sizes, class_names, device = load_data(data_dir)
    launch()




# https://github.com/pytorch/vision/issues/714
    # overfitting

# test on random images from the test set






# salloc -p gpu_test -t 0-01:00 --mem 16000 --gres=gpu:4
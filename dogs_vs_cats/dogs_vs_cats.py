
from glob import glob
from glob import iglob
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# path = 'D:\\PyTorch\\dogs_vs_cats\\data'
# files = glob(os.path.join(path, '*\\*.jpg'))
# print('总图片数: %s' % len(files))
# no_of_images = len(files)
# np.random.seed(0)
# shuffle = np.random.permutation(no_of_images)
# os.mkdir(os.path.join(path, 'valid'))

# for i in shuffle[:2000]:
#     folder = files[i].split('\\')[-1].split('.')[0]
#     image = files[i].split('\\')[-1]
#     os.rename(files[i], os.path.join(path, 'valid', folder, image))
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(        # mean和std的取值时随机的吗？
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train = ImageFolder('D:\\PyTorch\\dogs_vs_cats\\data\\train', simple_transform)
valid = ImageFolder('D:\\PyTorch\\dogs_vs_cats\\data\\valid', simple_transform)

print(train.class_to_idx)
print(train.classes)

train_data_gen = torch.utils.data.DataLoader(train, batch_size=64, num_workers=2)
valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=2)
print('###')
dataset_sizes = {
    'train': len(train_data_gen.dataset),
    'valid': len(valid_data_gen.dataset)
    }
dataloaders = {
    'train': train_data_gen,
    'valid': valid_data_gen
    }
model_ft = models.resnet18(pretrained=True)
num_dfts = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_dfts, 2)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(
    model_ft.parameters(),
    lr=learning_rate,
    momentum=0.9
)
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft,
    step_size=7,
    gamma=0.1
)


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    start_time = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in (:.0f)m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))  # 如此转置的原因？？？
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)


if __name__ == "__main__":
    train_model(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=5
    )

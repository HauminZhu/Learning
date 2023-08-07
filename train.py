import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


from model import resnet34, resnet50, ResNet, BasicBlock

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {"train" :  transforms.Compose([
                        # transforms.Resize(256),
                        # transforms.CenterCrop(224),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                    "test": transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    }

    data_folder = "/Users/hzhu/Documents/DeepLearning/flower_data"
    train_data_path = os.path.join(data_folder, "train")
    test_data_path = os.path.join(data_folder, "val")

    train_data = datasets.ImageFolder(root=train_data_path,
                                      transform=data_transform["train"])
    test_data = datasets.ImageFolder(root=test_data_path,
                                     transform=data_transform["test"])
    train_num = len(train_data)
    test_num = len(test_data)

    flower_class_to_idx = train_data.class_to_idx
    print(flower_class_to_idx)
    flower_idx_to_class = dict((val, key) for key, val in flower_class_to_idx.items())
    print(flower_idx_to_class)
    json_str = json.dumps(flower_idx_to_class, indent=4)
    with open("class_indices.json", 'w') as file:
        file.write(json_str)

    batch_size = 16
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("num_workers {}".format(num_workers))

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=num_workers)

    print("Using {} images for training, using {} images for testing".format(train_num, test_num))

    num_classes = 5
    net = resnet34(num_classes)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    num_epochs = 10
    save_path = "./resnet34_customized.pth"
    train_steps = len(train_loader)
    for epoch in range(num_epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            l = loss(logits, labels.to(device))
            l.backward()
            running_loss += l
            optimizer.step()
            train_bar.desc = "Train epoch[{}/{}], loss:{:.3f}".format(epoch + 1, num_epochs, l)

        # test/validation
        net.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for data in test_bar:
                images, labels = data
                logits = net(images.to(device))
                predict_y = torch.max(logits, dim=1)[1]
                acc += torch.eq(predict_y, labels.to(device)).sum().item()

                test_bar.desc = "Test epoch[{}/{}]".format(epoch + 1, num_epochs)
        test_acc = acc / test_num
        print("Epoch {}, training loss: {}, test accuracy {}".format(epoch + 1, running_loss, test_acc))






if __name__ == '__main__':
    main()

import torch
import torchvision # This library is used for image-based operations (Augmentations)
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import wandb
import matplotlib.pyplot as plt
from torchsummaryX import summary



# On Mar. 7th, the best result is obtained by ResNet[2,4,4,2] with fc_dropout(0.25) and batch_size=160 in the following manner:
# 0.1lr for 40 epochs, 0.01lr for 30 epochs, 0.001lr for 11 epochs, 0.0001lr for 3 epochs (with all transformations).
# The, comment off all transformations except normalization, with lr=0.001 for 5 epochs,
# gives the best classification val_acc=85.833% and the best verification val_acc=50.278%

def main():

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", DEVICE)

    config = {
        'Resume': False, # Set this to True if you want to resume a previous run
        'wb_watch': True, # Set this to True if you want to use wandb
        'train_partial': False, # Set this to True if you want to train on a subset of the data
        'batch_size': 160, # Increase this if your GPU can handle it, max 192
        'lr': 0.1,
        'steps': 5,
        'epochs': 80, # 20 epochs is recommended ONLY for the early submission - you will have to train for much longer typically.
        # Include other parameters as needed.
    }

    # Data paths
    DATA_DIR    = "/home/dunhanj/TestRuns/11-785-s24-hw2p2-classification"
    TRAIN_DIR   = os.path.join(DATA_DIR, "train")
    VAL_DIR     = os.path.join(DATA_DIR, "dev")
    TEST_DIR    = os.path.join(DATA_DIR, "test")

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        torchvision.transforms.RandomRotation(degrees=15),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomErasing(scale=(0.1, 0.1)),
        torchvision.transforms.Normalize(mean=[0.5102566480636597, 0.4014372229576111, 0.3508550226688385], 
                                         std=[0.30747994780540466, 0.2700912356376648, 0.25909197330474854])  # from ImageNet statistics
        ]) # TODO: Specify transformations/augmentations performed on the train dataset
    
    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5102566480636597, 0.4014372229576111, 0.3508550226688385], 
                                         std=[0.30747994780540466, 0.2700912356376648, 0.25909197330474854])  # from ImageNet statistics
        ]) # TODO: Specify transformations performed on the val dataset
    
    train_dataset   = torchvision.datasets.ImageFolder(TRAIN_DIR, transform = train_transforms)
    valid_dataset   = torchvision.datasets.ImageFolder(VAL_DIR, transform = valid_transforms)
    train_loader = torch.utils.data.DataLoader(dataset       = train_dataset,
                                               batch_size    = config['batch_size'],
                                               shuffle       = True,
                                               num_workers   = 16, # Uncomment this line if you want to increase your num workers
                                               pin_memory    = True
                                               )
    valid_loader = torch.utils.data.DataLoader(dataset       = valid_dataset,
                                               batch_size    = config['batch_size'],
                                               shuffle       = False,
                                               num_workers   = 8 # Uncomment this line if you want to increase your num workers
                                               )
    
    class TestDataset(torch.utils.data.Dataset):

        def __init__(self, data_dir, transforms):
            self.data_dir   = data_dir
            self.transforms = transforms

            # This one-liner basically generates a sorted list of full paths to each image in the test directory
            self.img_paths  = list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            return self.transforms(Image.open(self.img_paths[idx]))
    
    test_dataset = TestDataset(TEST_DIR, transforms = valid_transforms)
    test_loader = torch.utils.data.DataLoader(dataset     = test_dataset,
                                              batch_size  = config['batch_size'],
                                              shuffle     = False,
                                              drop_last   = False,
                                              num_workers = 8 # Uncomment this line if you want to increase your num workers
                                              )
    
    print("Number of classes    : ", len(train_dataset.classes))
    print("No. of train images  : ", train_dataset.__len__())
    print("Shape of image       : ", train_dataset[0][0].shape)
    print("Batch size           : ", config['batch_size'])
    print("Train batches        : ", train_loader.__len__())
    print("Val batches          : ", valid_loader.__len__())

    """
    class FeatureResNet18(torchvision.models.ResNet):
        def __init__(self, num_classes=7001):
            super(FeatureResNet18, self).__init__(block=torchvision.models.resnet.BasicBlock, layers=[2, 4, 4, 2], num_classes=num_classes)
            self.feature_extractor = torch.nn.Sequential(*list(self.children())[:-1])  # Exclude the last fully connected layer
        def forward(self, x):
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1)  # Flatten the features
            out = self.fc(features)  # Use the original fully connected layer for classification
            return features, out
    
    class Network(torch.nn.Module):
        def __init__(self, num_classes=7001):   # TODO: change the structure of the network 
            super().__init__()
            self.backbone = FeatureResNet18(num_classes=7001).to(DEVICE)
        def forward(self, x, return_feats=False):
            feats, out = self.backbone(x)
            if return_feats:
                return feats  # Return features from before the fully connected layer
            else:
                return out  # Return the final classification output
    """

    # Define a basic residual block
    class ResidualBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            # First convolutional layer
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU(inplace=True)
            # Second convolutional layer
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            # Shortcut connection if dimensions change
            self.shortcut = torch.nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(out_channels))
        def forward(self, x):
            residual = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x += self.shortcut(residual)
            x = self.relu(x)
            return x

    # Define the ResNet architecture
    class ResNet(torch.nn.Module):
        def __init__(self, block, num_blocks, num_classes=7001):
            super(ResNet, self).__init__()
            self.in_channels = 64
            # Initial convolutional layer
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(64)
            self.relu = torch.nn.ReLU(inplace=True)
            self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # Residual blocks
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            # Global average pooling and fully connected layer
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(512, num_classes)
            self.dropout = torch.nn.Dropout(p=0.25)
            """
            # weight initialization
            for l in self.modules():
                if isinstance(l, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(l.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(l, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(l.weight, 1)
                    torch.nn.init.constant_(l.bias, 0)
            """
        def _make_layer(self, block, out_channels, num_blocks, stride):
            layers = []
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(block(out_channels, out_channels, 1))
            return torch.nn.Sequential(*layers)

        def forward(self, x, return_feats=False):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            if return_feats:
                return x  # Return features from before the fully connected layer
            x = self.fc(x)
            x = self.dropout(x)
            return x  # Return the final classification output
    # """
        
    # Create a ResNet instance
    model = ResNet(ResidualBlock, [2, 4, 4, 2]).to(DEVICE)

    # Initialize your model
    # model = Network().to(DEVICE)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-3)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=5e-2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = torch.cuda.amp.GradScaler()


    def train(model, dataloader, optimizer, criterion):
        model.train()
        batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)
        num_correct = 0
        total_loss  = 0
        if config['train_partial']:
            j = 0
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad() # Zero gradients
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast(): # This implements mixed precision. Thats it!
                outputs = model(images)
                loss    = criterion(outputs, labels)
            num_correct     += int((torch.argmax(outputs, axis=1) == labels).sum())
            total_loss      += float(loss.item())
            batch_bar.set_postfix(
                acc         = "{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
                loss        = "{:.04f}".format(float(total_loss / (i + 1))),
                num_correct = num_correct,
                lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr']))
            )
            scaler.scale(loss).backward() # This is a replacement for loss.backward()
            scaler.step(optimizer) # This is a replacement for optimizer.step()
            scaler.update()
            batch_bar.update() # Update tqdm bar
            if config['train_partial']:
                j += 1
                if j > 50:
                    break

        batch_bar.close() # You need this to close the tqdm bar
        acc         = 100 * num_correct / (config['batch_size']* len(dataloader))
        total_loss  = float(total_loss / len(dataloader))
        return acc, total_loss

    def validate(model, dataloader, criterion):
        model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)
        num_correct = 0.0
        total_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # with torch.inference_mode(): ########## CHANGED: (conmmented off)
            outputs = model(images) ########## CHANGED: (indentation removed)
            loss = criterion(outputs, labels) ########## CHANGED: (indentation removed)
            num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
            total_loss += float(loss.item())
            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                num_correct=num_correct)
            batch_bar.update()
        batch_bar.close()
        acc = 100 * num_correct / (config['batch_size']* len(dataloader))
        total_loss = float(total_loss / len(dataloader))
        return acc, total_loss
    

    gc.collect() # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    # This obtains the list of known identities from the known folder
    known_regex = "/home/dunhanj/TestRuns/11-785-s24-hw2p2-verification/known/*/*"
    known_paths = [i.split('/')[-2] for i in sorted(glob.glob(known_regex))]
    # Obtain a list of images from unknown folders
    unknown_dev_regex = "/home/dunhanj/TestRuns/11-785-s24-hw2p2-verification/unknown_dev/*"
    unknown_test_regex = "/home/dunhanj/TestRuns/11-785-s24-hw2p2-verification/unknown_test/*"
    # We load the images from known and unknown folders
    unknown_dev_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_dev_regex)))]
    unknown_test_images = [Image.open(p) for p in tqdm(sorted(glob.glob(unknown_test_regex)))]
    known_images = [Image.open(p) for p in tqdm(sorted(glob.glob(known_regex)))]
    # Why do you need only ToTensor() here?
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5102566480636597, 0.4014372229576111, 0.3508550226688385], 
                                         std=[0.30747994780540466, 0.2700912356376648, 0.25909197330474854])
        ])
    unknown_dev_images = torch.stack([transforms(x) for x in unknown_dev_images])
    unknown_test_images = torch.stack([transforms(x) for x in unknown_test_images])
    known_images  = torch.stack([transforms(y) for y in known_images ])
    # Print your shapes here to understand what we have done
    # You can use other similarity metrics like Euclidean Distance if you wish
    similarity_metric = torch.nn.CosineSimilarity(dim=-1, eps= 1e-6) 


    def eval_verification(unknown_images, known_images, model, similarity, batch_size= config['batch_size'], mode='val'):
        unknown_feats, known_feats = [], []
        batch_bar = tqdm(total=len(unknown_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
        model.eval()
        # We load the images as batches for memory optimization and avoiding CUDA OOM errors
        for i in range(0, unknown_images.shape[0], batch_size):
            unknown_batch = unknown_images[i:i+batch_size] # Slice a given portion upto batch_size
            with torch.no_grad():
                unknown_feat = model(unknown_batch.float().to(DEVICE), return_feats=True) #Get features from model
            unknown_feats.append(unknown_feat)
            batch_bar.update()
        batch_bar.close()
        batch_bar = tqdm(total=len(known_images)//batch_size, dynamic_ncols=True, position=0, leave=False, desc=mode)
        for i in range(0, known_images.shape[0], batch_size):
            known_batch = known_images[i:i+batch_size]
            with torch.no_grad():
                known_feat = model(known_batch.float().to(DEVICE), return_feats=True)
            known_feats.append(known_feat)
            batch_bar.update()
        batch_bar.close()
        # Concatenate all the batches
        unknown_feats = torch.cat(unknown_feats, dim=0)
        known_feats = torch.cat(known_feats, dim=0)
        similarity_values = torch.stack([similarity(unknown_feats, known_feature) for known_feature in known_feats])
        # Print the inner list comprehension in a separate cell - what is really happening?
        max_similarity_values, predictions = similarity_values.max(0) #Why are we doing an max here, where are the return values?
        max_similarity_values, predictions = max_similarity_values.cpu().numpy(), predictions.cpu().numpy()
        # Note that in unknown identities, there are identities without correspondence in known identities.
        # Therefore, these identities should be not similar to all the known identities, i.e. max similarity will be below a certain
        # threshold compared with those identities with correspondence.
        # In early submission, you can ignore identities without correspondence, simply taking identity with max similarity value
        pred_id_strings = [known_paths[i] for i in predictions] # Map argmax indices to identity strings
        # After early submission, remove the previous line and uncomment the following code
        # threshold = # Choose a proper threshold
        # NO_CORRESPONDENCE_LABEL = 'n000000'
        # pred_id_strings = []
        # for idx, prediction in enumerate(predictions):
        #     if max_similarity_values[idx] < threshold: # why < ? Thank about what is your similarity metric
        #         pred_id_strings.append(NO_CORRESPONDENCE_LABEL)
        #     else:
        #         pred_id_strings.append(known_paths[prediction])
        if mode == 'val':
            true_ids = pd.read_csv('/home/dunhanj/TestRuns/11-785-s24-hw2p2-verification/verification_dev.csv')['label'].tolist()
            accuracy = 100 * accuracy_score(pred_id_strings, true_ids)
            #print("Verification Accuracy = {}".format(accuracy))
            return accuracy, pred_id_strings
        elif mode == 'test':
            return pred_id_strings


    gc.collect() # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    if config['wb_watch']:
        # weight and bias login
        wandb.login(key="e82cd60c71ce53e010026113443de725b0d4fb58") # API Key is in your wandb account, under settings (wandb.ai/settings)
        # Create your wandb run
        run = wandb.init(
            name = "early-submission", ## Wandb creates random run names if you skip this field
            reinit = True, ### Allows reinitalizing runs when you re-run this cell
            # run_id = ### Insert specific run id here if you want to resume a previous run
            # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
            project = "hw2p2-ablations", ### Project should be created in your wandb account
            config = config ### Wandb Config for your run
        )
        wandb.watch(model, log="all")

    if config['Resume']:
        checkpoint = torch.load("/home/dunhanj/TestRuns/checkpoint_verification.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint")
    
    best_class_acc      = 0.0
    best_ver_acc        = 0.0
    for epoch in range(config['epochs']):
        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_acc, train_loss = train(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion) # TODO
        print("\nEpoch {}/{}: \nTrain Acc (Classification) {:.04f}% Train Loss (Classification) {:.04f} Learning Rate {:.04f}".format(epoch + 1, config['epochs'], train_acc, train_loss, curr_lr))
        val_acc, val_loss = validate(model, valid_loader, criterion) # TODO
        ver_acc, _ = eval_verification(unknown_dev_images, known_images, model, similarity_metric, config['batch_size'], mode='val')
        print("Val Acc (Classification) {:.04f}% Val Loss (Classification) {:.04f}% Val Acc (Verification) {:.04f}%".format(val_acc, val_loss, ver_acc))
        scheduler.step(val_loss)
        # wandb.log record
        if config['wb_watch']:
            wandb.log({"train_classification_acc": train_acc,
                    "train_classification_loss":train_loss,
                    "val_classification_acc": val_acc,
                    "val_classification_loss": val_loss,
                    "val_verification_acc": ver_acc,
                    "learning_rate": curr_lr})
        # save the best classification model if necessary
        if val_acc >= best_class_acc:
            best_class_acc = val_acc
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch}, './checkpoint_classification.pth')
            if config['wb_watch']:
                wandb.save('checkpoint_classification.pth')
        # save the best verification model if necessary
        if ver_acc >= best_ver_acc:
            best_ver_acc = ver_acc
            torch.save({'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'val_acc': ver_acc,
                        'epoch': epoch}, './checkpoint_verification.pth')
            if config['wb_watch']:
                wandb.save('checkpoint_verification.pth')
        

    if config['wb_watch']:
        ### Finish your wandb run
        wandb.unwatch()
        run.finish()


    def test(model, dataloader): # TODO: Run to finish predicting on the test set.
        model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
        test_results = []
        for i, (images) in enumerate(dataloader):
            images = images.to(DEVICE)
            # with torch.inference_mode(): ########## CHANGED: (conmmented off)
            outputs = model(images) ########## CHANGED: (indentation removed)
            outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
            test_results.extend(outputs)
            batch_bar.update()
        batch_bar.close()
        return test_results


    # for classification submission, load the best model for classification and then run test
    checkpoint = torch.load("/home/dunhanj/TestRuns/checkpoint_classification.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_results = test(model, test_loader)
    with open("classification_submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(test_dataset)):
            f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", test_results[i]))

    # for verification submission, load the best model for verification and then run eval_verification
    checkpoint = torch.load("/home/dunhanj/TestRuns/checkpoint_verification.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    pred_id_strings = eval_verification(unknown_test_images,
                                        known_images,
                                        model,
                                        similarity_metric,
                                        config['batch_size'],
                                        mode='test')
    with open("verification_submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(pred_id_strings)):
            f.write("{},{}\n".format(i, pred_id_strings[i]))
    print("new submission files created")


if __name__ == "__main__":
    main()



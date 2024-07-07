import torch
import numpy as np
from torchsummaryX import summary
# import sklearn
import gc
# import zipfile
# import pandas as pd
from tqdm.auto import tqdm
import os
# import datetime
import wandb


def main():
    print("Hello World!")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)



    # Wandb Initialization
    wandb.login(key="e82cd60c71ce53e010026113443de725b0d4fb58") #API Key is in your wandb account, under settings (wandb.ai/settings)



    # MLP Model Architecture
    # However, you need to run a lot of experiments to cross the medium or high cutoff
    # trial 2: diamond_deep: [1024，1024，1024，1300，1500，1700，1900，2048，2048，1024，512，512，512，512], 
    #          with 0.23 dropout per layer, 5 epochs finished, 80.5% validation accuracy.
    # trial 3: diamond_less_deep: [1100, 1300, 1500, 1700, 1900, 2100, 2300, 1700, 480], 
    #          with no dropout, 2 epochs finished, 81.87% validation accuracy.
    # trial 4: diamond_less_deep: [1100, 1300, 1500d, 1700d, 1900d, 2100d, 2300d, 1700, 480],
    #          with 0.20 dropout, 3 epochs finished, 81.96% validation accuracy.
    # trial 5: diamond_less_deep: [1100d, 1300d, 1500d, 1700d, 1900d, 2100d, 2300d, 1700, 480],
    #          with 0.23 dropout, 15 epochs finished, 84.46% validation accuracy.
    # trial 6: cylinder_deep: [1700d, 1700d, 1700d, 1700d, 1700d, 1700d, 1700d, 1700d, 1000d],
    #          with 0.24 dropout, 30 epochs finished, 85.45% validation accuracy.
    # trial 7: cylinder_deep: [2048d, 2048d, 2048d, 2048d, 2048d, 2048d, 2048], num_params: 23914538 (23.9M < 24M),
    #          with 0.20 dropout, 30 epochs finished, 86.38% validation accuracy.
    #          lr scheduler: 22 epochs of lr=0.01 (val_acc: 83.4916%), 8 epochs of lr=0.001 (val_acc: 86.0869%)
    #                        6 epochs of lr=0.0001 (val_acc: 86.3793%), 2 epochs of lr=0.00001 (val_acc: 86.39%)
    class Network(torch.nn.Module):

        def __init__(self, input_size, output_size):
            super(Network, self).__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_size, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Mish(),
                torch.nn.Dropout(0.20),

                torch.nn.Linear(2048, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Mish(),
                torch.nn.Dropout(0.20),

                torch.nn.Linear(2048, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Mish(),
                torch.nn.Dropout(0.20),

                torch.nn.Linear(2048, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Mish(),
                torch.nn.Dropout(0.20),

                torch.nn.Linear(2048, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Mish(),
                torch.nn.Dropout(0.20),

                torch.nn.Linear(2048, 2048),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Mish(),
                torch.nn.Dropout(0.20),

                torch.nn.Linear(2048, output_size)
            )

        def forward(self, x):
            out = self.model(x)
            return out



    ### PHONEME LIST
    PHONEMES = [
                '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',
                'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
                'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
                'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
                'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
                'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]']



    # Dataset class to load train and validation data
    class AudioDataset(torch.utils.data.Dataset):

        def __init__(self, root, phonemes = PHONEMES, context=0, partition= "train-clean-100"): # Feel free to add more arguments

            self.context    = context
            self.phonemes   = phonemes

            # TODO: MFCC directory - use partition to acces train/dev directories from kaggle data using root
            self.mfcc_dir       = root + '/' + partition + '/mfcc'
            # TODO: Transcripts directory - use partition to acces train/dev directories from kaggle data using root
            self.transcript_dir = root + '/' + partition + '/transcript'

            # TODO: List files in sefl.mfcc_dir using os.listdir in sorted order
            mfcc_names          = sorted(os.listdir(self.mfcc_dir))
            # TODO: List files in self.transcript_dir using os.listdir in sorted order
            transcript_names    = sorted(os.listdir(self.transcript_dir))

            # Making sure that we have the same no. of mfcc and transcripts
            assert len(mfcc_names) == len(transcript_names)

            self.mfccs, self.transcripts = [], []

            # TODO: Iterate through mfccs and transcripts
            for i in tqdm(range(len(mfcc_names)), desc="Loading TrainData..."):
            # for i in range(100):

                assert mfcc_names[i] == transcript_names[i] # Making sure that the filenames match

                #   Load a single mfcc
                #   Do Cepstral Normalization of mfcc (explained in writeup)
                mfcc_raw    = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
                self.mfccs.append((mfcc_raw - np.mean(mfcc_raw, axis=0)) / (np.std(mfcc_raw, axis=0) + 1e-8))
                #   Load the corresponding transcript
                #   (Is there an efficient way to do this without traversing through the transcript?)
                #   Note that SOS will always be in the starting and EOS at end, as the name suggests.
                #   Remove [SOS] and [EOS] from the transcript
                self.transcripts.append(np.load(os.path.join(self.transcript_dir, transcript_names[i]))[1 : -1])

            # NOTE:
            # Each mfcc is of shape T1 x 27, T2 x 27, ...
            # Each transcript is of shape (T1+2) x 27, (T2+2) x 27 before removing [SOS] and [EOS]

            # TODO: Concatenate all mfccs in self.mfccs such that
            # the final shape is T x 27 (Where T = T1 + T2 + ...)
            self.mfccs          = np.concatenate(self.mfccs, axis=0)

            # TODO: Concatenate all transcripts in self.transcripts such that
            # the final shape is (T,) meaning, each time step has one phoneme output
            self.transcripts    = np.concatenate(self.transcripts, axis=0)
            # Hint: Use numpy to concatenate

            # Length of the dataset is now the length of concatenated mfccs/transcripts
            self.length = len(self.mfccs)
            
            # Take some time to think about what we have done.
            # self.mfcc is an array of the format (Frames x Features).
            # Our goal is to recognize phonemes of each frame
            # From hw0, you will be knowing what context is.
            # We can introduce context by padding zeros on top and bottom of self.mfcc
            self.mfccs = np.pad(self.mfccs, pad_width=((self.context, self.context), (0, 0))) # TODO
            # print(self.mfccs.shape)
            # The available phonemes in the transcript are of string data type
            # But the neural network cannot predict strings as such.
            # Hence, we map these phonemes to integers

            # TODO: Map the phonemes to their corresponding list indexes in self.phonemes
            phoneme_to_index = {p:i for i, p in enumerate(self.phonemes)}
            t = []
            for i in self.transcripts:
                t.append(phoneme_to_index[str(i)])
            self.transcripts = t
            # Now, if an element in self.transcript is 0, it means that it is 'SIL' (as per the above example)
            # print(self.transcripts.shape)

        def __len__(self):
            return self.length

        def __getitem__(self, ind):

            # TODO: Based on context and offset, return a frame at given index with context frames to the left, and right.
            # After slicing, you get an array of shape 2*context+1 x 27. But our MLP needs 1d data and not 2d.
            frames      = torch.FloatTensor(self.mfccs[ind : (ind+2*self.context+1)]).flatten()
            phonemes    = torch.tensor(self.transcripts[ind])

            return frames, phonemes



    # Dataset class for test set
    class AudioTestDataset(torch.utils.data.Dataset):
        def __init__(self, root, context=0, partition= "test-clean"): # Feel free to add more arguments

            self.context    = context

            # TODO: MFCC directory - use partition to acces train/dev directories from kaggle data using root
            self.mfcc_dir       = root + '/' + partition + '/mfcc'

            # TODO: List files in sefl.mfcc_dir using os.listdir in sorted order
            mfcc_names          = sorted(os.listdir(self.mfcc_dir))

            self.mfccs = []

            # TODO: Iterate through mfccs and transcripts
            for i in tqdm(range(len(mfcc_names))):
            # for i in range(100):

                #   Load a single mfcc
                #   Do Cepstral Normalization of mfcc (explained in writeup)
                mfcc_raw    = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
                self.mfccs.append((mfcc_raw - np.mean(mfcc_raw, axis=0)) / (np.std(mfcc_raw, axis=0) + 1e-8))

            # NOTE:
            # Each mfcc is of shape T1 x 27, T2 x 27, ...
            # Each transcript is of shape (T1+2) x 27, (T2+2) x 27 before removing [SOS] and [EOS]

            # TODO: Concatenate all mfccs in self.mfccs such that
            # the final shape is T x 27 (Where T = T1 + T2 + ...)
            self.mfccs          = np.concatenate(self.mfccs, axis=0)
            # Hint: Use numpy to concatenate

            # Length of the dataset is now the length of concatenated mfccs/transcripts
            self.length = len(self.mfccs)
            self.mfccs = np.pad(self.mfccs, pad_width=((self.context, self.context), (0, 0))) # TODO

        def __len__(self):
            return self.length

        def __getitem__(self, ind):

            # TODO: Based on context and offset, return a frame at given index with context frames to the left, and right.
            # After slicing, you get an array of shape 2*context+1 x 27. But our MLP needs 1d data and not 2d.
            frames = torch.FloatTensor(self.mfccs[ind : (ind+2*self.context+1)]).flatten()

            return frames

        # TODO: Create a test dataset class similar to the previous class but you dont have transcripts for this
        # Imp: Read the mfccs in sorted order, do NOT shuffle the data here or in your dataloader.



    # Hyperparameters
    config = {
        'epochs'        : 50,
        'batch_size'    : 1024*4,
        'context'       : 25,
        'init_lr'       : 0.01,
        'architecture'  : 'diamond_6layers'
        # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
    }



    # Create Dataset
    # PATH to local dataset (currently partial training dataset for testing and debugging purposes)
    root = "/home/dunhanj/TestRuns/11-785-s24-hw1p2"
    #TODO: Create a dataset object using the AudioDataset class for the training data
    train_data = AudioDataset(root, PHONEMES, context=config['context'], partition="train-clean-100")
    print("train_data loaded")
    # TODO: Create a dataset object using the AudioDataset class for the validation data
    val_data = AudioDataset(root, PHONEMES, context=config['context'], partition="dev-clean")
    print("val_data loaded")
    # TODO: Create a dataset object using the AudioTestDataset class for the test data
    test_data = AudioTestDataset(root, context=config['context'], partition="test-clean")
    print("test_data loaded")


    # Define dataloaders for train, val and test datasets
    # Dataloaders will yield a batch of frames and phonemes of given batch_size at every iteration
    # We shuffle train dataloader but not val & test dataloader. Why?
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data,
        num_workers = 32, # 4
        batch_size  = config['batch_size'],
        pin_memory  = True,
        shuffle     = True
    )
    print("train_loader loaded")
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data,
        num_workers = 2, # 2
        batch_size  = config['batch_size'],
        pin_memory  = True,
        shuffle     = False
    )
    print("val_loader loaded")
    test_loader = torch.utils.data.DataLoader(
        dataset     = test_data,
        num_workers = 2, # 2
        batch_size  = config['batch_size'],
        pin_memory  = True,
        shuffle     = False
    )
    print("test_loader loaded")
    print("Batch size     : ", config['batch_size'])
    print("Context        : ", config['context'])
    print("Input size     : ", (2*config['context']+1)*27)
    print("Output symbols : ", len(PHONEMES))
    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    print("Validation dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
    print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))



    # Testing code to check if your data loaders are working
    for i, data in enumerate(train_loader):
        frames, phoneme = data
        print(frames.shape, phoneme.shape)
        break



    # Define Model, Loss Function, Optimizer and optionally a Learning Rate Scheduler
    INPUT_SIZE  = (2*config['context'] + 1) * 27 # Why is this the case?
    model       = Network(INPUT_SIZE, len(train_data.phonemes)).to(device)
    print(INPUT_SIZE, len(train_data.phonemes))
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    # summary(model, frames.to(device))
    criterion = torch.nn.CrossEntropyLoss() # Defining Loss function.
    # We use CE because the task is multi-class classification
    optimizer = torch.optim.AdamW(model.parameters(), lr= config['init_lr']) #Defining Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=5e-2)
    # Recommended : Define Scheduler for Learning Rate,
    # including but not limited to StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, etc.
    # You can refer to Pytorch documentation for more information on how to use them.
    # Is your training time very high?
    # Look into mixed precision training if your GPU (Tesla T4, V100, etc) can make use of it
    # Refer - https://pytorch.org/docs/stable/notes/amp_examples.html



    torch.cuda.empty_cache()
    gc.collect()



    def train(model, dataloader, optimizer, criterion):

        model.train()
        tloss, tacc = 0, 0 # Monitoring loss and accuracy
        batch_bar   = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        for i, (frames, phonemes) in enumerate(dataloader):

            ### Initialize Gradients
            optimizer.zero_grad()
            ### Move Data to Device (Ideally GPU)
            frames      = frames.to(device)
            phonemes    = phonemes.to(device)
            ### Forward Propagation
            logits  = model(frames)
            ### Loss Calculation
            loss    = criterion(logits, phonemes)
            ### Backward Propagation
            loss.backward()
            ### Gradient Descent
            optimizer.step()
            tloss   += loss.item()
            tacc    += torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]
            batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                                acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
            batch_bar.update()
            ### Release memory
            del frames, phonemes, logits
            torch.cuda.empty_cache()

        batch_bar.close()
        tloss   /= len(train_loader)
        tacc    /= len(train_loader)
        return tloss, tacc



    def eval(model, dataloader, criterion, device):

        model.eval() # set model in evaluation mode
        vloss, vacc = 0, 0 # Monitoring loss and accuracy
        batch_bar   = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

        for i, (frames, phonemes) in enumerate(dataloader):

            ### Move data to device (ideally GPU)
            frames      = frames.to(device)
            phonemes    = phonemes.to(device)

            # makes sure that there are no gradients computed as we are not training the model now
            with torch.inference_mode():
                ### Forward Propagation
                logits  = model(frames)
                ### Loss Calculation
                loss    = criterion(logits, phonemes)

            vloss   += loss.item()
            vacc    += torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]

            # Do you think we need loss.backward() and optimizer.step() here?

            batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                                acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
            batch_bar.update()

            ### Release memory
            del frames, phonemes, logits
            torch.cuda.empty_cache()

        batch_bar.close()
        vloss   /= len(val_loader)
        vacc    /= len(val_loader)

        return vloss, vacc



    # Create your wandb run
    run = wandb.init(
        name    = "cylinder_6layers", ### Wandb creates random run names if you skip this field, we recommend you give useful names
        reinit  = True, ### Allows reinitalizing runs when you re-run this cell
        #id     = "y28t31uz", ### Insert specific run id here if you want to resume a previous run
        #resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "hw1p2", ### Project should be created in your wandb account
        config  = config ### Wandb Config for your run
    )

    ### Save your model architecture as a string with str(model)
    model_arch  = str(model)

    ### Save it in a txt file
    arch_file   = open("model_arch.txt", "w")
    file_write  = arch_file.write(model_arch)
    arch_file.close()

    ### log it in your wandb run with wandb.save()
    wandb.save('model_arch.txt')



    # Load model from checkpoint if needed
    load_model = False
    if load_model:
        checkpoint = torch.load('./mlp_models/bestacc8639.pth')
        model.load_state_dict(checkpoint['model_state_dict'])



    # Iterate over number of epochs to train and evaluate your model
    torch.cuda.empty_cache()
    gc.collect()
    wandb.watch(model, log="all")
    best_val_acc = 0
    for epoch in range(config['epochs']):

        print("\nEpoch {}/{}".format(epoch+1, config['epochs']))
        curr_lr                 = float(optimizer.param_groups[0]['lr'])
        train_loss, train_acc   = train(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion)
        val_loss, val_acc       = eval(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, curr_lr))
        print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc*100, val_loss))
        ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
        # Optionally, you can log at each batch inside train/eval functions
        # (explore wandb documentation/wandb recitation)
        wandb.log({'train_acc': train_acc*100, 'train_loss': train_loss,
                'val_acc': val_acc*100, 'valid_loss': val_loss, 'lr': curr_lr})

        ### Highly Recommended: Save checkpoint in drive and/or wandb if accuracy is better than your current best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}, "/home/dunhanj/TestRuns/mlp_models/best_acc.pth")
            
            print("\tNew Model Checkpoint-Best Val Acc {:.04f}%\tBest Val Epoch {}".format(best_val_acc, best_epoch))

    ### Finish your wandb run
    wandb.unwatch()
    run.finish()



    def test(model, test_loader):
        ### What you call for model to perform inference?
        model.eval # TODO train or eval?

        ### List to store predicted phonemes of test data
        test_predictions = []

        ### Which mode do you need to avoid gradients?
        with torch.no_grad(): # TODO

            for i, mfccs in enumerate(tqdm(test_loader)):

                mfccs   = mfccs.to(device)

                logits  = model(mfccs)

                ### Get most likely predicted phoneme with argmax
                predicted_phonemes = np.argmax(logits.cpu().numpy(), axis=1) # TODO

                ### How do you store predicted_phonemes with test_predictions? Hint, look at eval
                test_predictions.extend(predicted_phonemes)

        return test_predictions


    predictions = test(model, test_loader)
    for i in range(len(predictions)):
        predictions[i] = PHONEMES[predictions[i]]


    ### Create CSV file with predictions
    with open("./submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(predictions)):
            f.write("{},{}\n".format(i, predictions[i]))





if __name__ == "__main__":
    main()

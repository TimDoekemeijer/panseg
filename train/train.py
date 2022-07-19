'''
    TRAIN
    File to train and evaluate on the dataset.
    This script will output per fold: learning curves, best-performing model, prediction (in a .nii.gz file).

    @author: tdoekemeijer
'''

'''
Structure in dataset:
dirName
├── MATRIX/
│   ├── Images/
│   │   ├── MATRIX_Phase1_2_DWI_MASK_ADC600_100.nii.gz
│   │   └── MATRIX_Phase1_3_DWI_MASK_ADC600_100.nii.gz
│   └── Masks/
│       ├── MATRIX_Phase1_2_DWI_MASK_ADC600_100-label.nii.gz
│       └── MATRIX_Phase1_3_DWI_MASK_ADC600_100_1-label.nii.gz
├── REPRO/
│   ├── ...
└── CR/
    ├── ...

Structure in saved files:
SLURMoutput/
├── graphs/
├── models/
└── prednif/

'''
dirName = '/.../PADDED/'        #'/.../PADDED_CROP/' for cropped files
num_epochs = 100
early_stopping = 10
lr = 0.028
nr_kfolds = 61                  #LOOCV -> nr. samples - 1
val_size = 0.1
random_seed = 42
train_batch_size = 16
test_batch_size = 1
print(f'Parameters:\n Epochs: {num_epochs}\n Early stopping:{early_stopping}\n Learning rate: {lr}\n K-folds:{nr_kfolds}\n Val_size:{val_size}\n')


from train.utils import *
from train.loss_functions import *
from misc.getLists import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets
from medicaltorch import models as mt_models
from medicaltorch import losses as mt_losses
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for  PyTorch RNG, Python RNG, numpy RNG
# Brings consistency between runs
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Start time to keep track on the duration of the training and evaluation process
start = time.time()

# Get lists of images and corresponding masks
listOfImages = getListOfPPImages(dirName)
listOfMasks = getListOfPPMasks(dirName)
filename_pairs = get_filename_pairs(listOfImages, listOfMasks)
print(filename_pairs, len(filename_pairs), "\n")

# Make use of GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\ndevice:', device, torch.cuda.current_device())

# Make a list of unique patients. Filenames are determined with set indices.
listpat = []
for each_pair in filename_pairs:
    for one in each_pair:
        if "Images/CR_CR" in one:
            if dirName == '/.../PADDED/':
                listpat.append(one[51:55])
            if dirName == '/.../PADDED_CROP/':
                listpat.append(one[56:60])
        if "Images/MATRIX" in one:
            if dirName == '/.../PADDED/':
                listpat.append(one[59:68])
            if dirName == '/.../PADDED_CROP/':
                listpat.append(one[64:73])
        if "Images/REPRO" in one:
            if dirName == '/.../PADDED/':
                listpat.append(one[51:59])
            if dirName == '/.../PADDED_CROP/':
                listpat.append(one[56:64])
patients = np.unique(listpat)

print('------------------------------')

# Define preprocessing transformations.
preprocessing_transforms = torchvision.transforms.Compose([
    HistogramClipping(),
    RangeNorm(),
    mt_transforms.ToTensor(),
])

# Define augmentation transformations for training portion.
rot_degree = 10
transl_range = [0.05,0.05]
shear_range = [-5, 5]

train_transforms = torchvision.transforms.Compose([
    mt_transforms.RandomRotation(rot_degree),
    mt_transforms.RandomAffine(0, translate=transl_range),
    mt_transforms.RandomAffine(0, shear=shear_range),
    mt_transforms.ToTensor(),
    ])


# Create a dictionary to store fold results.
results = {}

# Define cross validator in order of patients.
kfold = KFold(n_splits=nr_kfolds,
              shuffle=False,
              )

# Start training per fold
print('STARTING TRAINING AND K-FOLD CROSS-VALIDATION')

for fold, (test_index, train_index) in enumerate(kfold.split(patients)):
    print(f'FOLD {fold + 1} of {nr_kfolds}')
    print('------------------------------')

    # Split unique patients in training and testing portion, according to the K-fold cross-validation split.
    # These patients are linked with their original filenames in the function.
    test_files, train_files_k = sub_filename_pairs(patients, test_index, train_index, listpat, filename_pairs)

    # Make a list of unique patients in training portion and split these randomly in training and validation portions.
    # These patients are linked with their original filenames, using a Boolean list.
    listpat_train_k = []
    for each_pair in train_files_k:
        for one in each_pair:
            if "Images/CR_CR" in one:
                if dirName == '/.../PADDED/':
                    listpat_train_k.append(one[51:55])
                if dirName == '/.../PADDED_CROP/':
                    listpat_train_k.append(one[56:60])
            if "Images/MATRIX" in one:
                if dirName == '/.../PADDED/':
                    listpat_train_k.append(one[59:68])
                if dirName == '/.../PADDED_CROP/':
                    listpat_train_k.append(one[64:73])
            if "Images/REPRO" in one:
                if dirName == '/.../PADDED/':
                    listpat_train_k.append(one[51:59])
                if dirName == '/.../PADDED_CROP/':
                    listpat_train_k.append(one[56:64])
    unique_train = np.unique(listpat_train_k)
    train_pat, val_pat = train_test_split(unique_train, test_size=val_size, shuffle=True, random_state=random_seed)
    blist = []
    for entry in listpat_train_k:
        if entry in train_pat:
            blist.append(True)
        if entry in val_pat:
            blist.append(False)
    train_files = []
    val_files = []
    for i in range(len(blist)):
        if blist[i] == True:
            train_files.append(train_files_k[i])
        if blist[i] == False:
            val_files.append(train_files_k[i])

    print(f'train/validation/test volume numbers= {len(train_files)}/ {len(val_files)}/ {len(test_files)}, split in: '
          f'{len(train_files) / len(filename_pairs)}, {len(val_files) / len(filename_pairs)}, {len(test_files) / len(filename_pairs)}\n')

    # Define datasets per portion.
    print('start preparing datasets')
    train_dataset = MRI2DSegmentationDataset(train_files,
                                             preprocess=preprocessing_transforms,
                                             transform=train_transforms,
                                             slice_filter_fn=slice_filtering_count0,
                                             )

    val_dataset = MRI2DSegmentationDataset(val_files,
                                           preprocess=preprocessing_transforms,
                                           slice_filter_fn=slice_filtering_count0,
                                           )

    test_dataset = MRI2DSegmentationDataset(test_files,
                                            preprocess=preprocessing_transforms,
                                            slice_filter_fn=slice_filtering_count0,
                                            )

    # Define dataloaders per portion.
    print('start preparing dataloaders')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  collate_fn=mt_datasets.mt_collate)
    print(f'Number of slices training:{len(train_dataloader.dataset)}')

    val_dataloader = DataLoader(val_dataset,
                                batch_size=train_batch_size,
                                shuffle=True,
                                collate_fn=mt_datasets.mt_collate)
    print(f'Number of slices validation:{len(val_dataloader.dataset)}')

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 collate_fn=mt_datasets.mt_collate)
    print(f'Number of slices test:{len(test_dataloader.dataset)}')

    totalslices = len(train_dataloader.dataset) + len(val_dataloader.dataset) + len(test_dataloader.dataset)
    print(f'train/validation/test slice numbers= {len(train_dataloader.dataset)}/ {len(val_dataloader.dataset)}/ {len(test_dataloader.dataset)}, split in: '
          f'{len(train_dataloader.dataset) / totalslices}, {len(val_dataloader.dataset) / totalslices}, {len(test_dataloader.dataset) / totalslices}')

    # Load U-net and send to GPU.
    network = mt_models.Unet()
    network.apply(reset_weights)
    network.to(device)

    # Initialize optimizer and criterion
    optimizer = optim.Adamax(network.parameters(), lr=lr)
    criterion = DiceLoss()              #FocalLoss() or #DiceBCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    # Start training loop
    print('start training loop')
    val_losses = []
    train_losses = []

    early_stopping_counter = 0
    best_loss = 1
    for epoch in range(0, num_epochs):

        print(f'Training epoch {epoch + 1}')
        network.train()
        train_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader)):
            images, labels = batch["input"], batch["gt"]
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = network(images)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        print(f'Validating epoch {epoch + 1}')
        network.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                images, labels = batch['input'], batch['gt']
                images, labels = images.to(device), labels.to(device)

                preds = network(images)
                loss = criterion(preds, labels)
                val_loss += loss.item()


        # Make lists of mean training and validation loss per epoch
        mean_train_loss = train_loss / len(train_dataloader)
        print('mean train loss:', mean_train_loss)
        train_losses.append(mean_train_loss)

        mean_val_loss = val_loss / len(val_dataloader)
        print('mean validation loss', mean_val_loss)
        val_losses.append(mean_val_loss)

        scheduler.step(mean_val_loss)

        # Save mean training and validation loss figures
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(val_losses, label="validation loss")
        plt.plot(train_losses, label="training loss")
        plt.xlabel("Epochs")
        plt.ylabel("Dice loss")
        plt.legend()
        plt.savefig(f'/.../SLURMoutput/graphs/loss-fold-{fold + 1}.png')

        # Early stopping process
        if epoch % 5 == 0:
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                save_path = f'/.../SLURMoutput/models/model-fold-{fold + 1}.pth'
                torch.save(network.state_dict(), save_path)
                print(f"Model updated")
            else:
                early_stopping_counter += 1
        print('best mean val loss', best_loss)
        print(f'Early stopping counter: {early_stopping_counter}')

        if early_stopping_counter == early_stopping:
            print('Early stopping')
            break

    # Get best UNET model per fold and send it to GPU.
    PATH = f'/.../SLURMoutput/models/model-fold-{fold + 1}.pth'

    testnetwork = mt_models.Unet()
    testnetwork.load_state_dict(torch.load(PATH))
    testnetwork.to(device)

    # Testing process
    print('Start testing')
    testnetwork.eval()
    total_score = 0
    preds = []

    # Evaluation for current fold
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images, labels = batch['input'], batch['gt']
            images, labels = images.to(device), labels.to(device)

            # Predictions
            pred = testnetwork(images)

            # Calculate Dice Similarity Coefficient per slice
            dsc = -1 * mt_losses.dice_loss(pred, labels)
            print('Dice score:', dsc.item())
            total_score += dsc.item()

            # Send predictions to cpu and make list of predictions
            pred2 = pred.cpu().detach().numpy()
            preds.append(np.reshape(pred2, [160, 64, 1])) #[128, 32, 1] for cropped

        print(f'Dice score for fold {fold + 1}:', total_score / len(test_dataloader))
        print('------------------------------')
        results[fold + 1] = total_score / len(test_dataloader)

    # Make NIfTI-files out of predictions
    prediction_stack = np.stack(preds, axis=2)
    prediction_stack = np.array(prediction_stack, dtype=np.int16)

    # Get aff of a certain volume
    aff = nib.load('/.../filename.nii.gz')
    prednif = nib.Nifti1Image(prediction_stack, aff.affine)
    nib.save(prednif, f'/.../SLURMoutput/prednif/Prediction{fold + 1}.nii.gz')

# Print all results and statistics per fold
k_folds = nr_kfolds
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('------------------------------')
sum_res = []
for key, value in results.items():
    print(f'Fold {key}: {value}')
    sum_res.append(value)

mean = np.mean(sum_res)
median = np.median(sum_res)
std = np.std(sum_res)
min = np.min(sum_res)
max = np.max(sum_res)

print('Mean:', mean)
print('Median:', median)
print('Std:', std)
print('Min:', min)
print('Max:', max)

# Print elapsed time for script.
elapsed = (time.time() - start)
print("\nElapsed time: " + time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:11], time.gmtime(elapsed)))
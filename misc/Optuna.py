"""
    OPTUNA
    Hyperparameter optimization for learning rate and optimizer.
    For description per line, see train.py

    @author: tdoekemeijer
"""

dirName = '/.../PADDED/'
num_epochs = 40
n_trials = 50

early_stopping = 10
test_size = 0.10
random_seed = 42
train_batch_size = 16

from train.utils import *
from train.loss_functions import *
from misc.getLists import *
import optuna
from optuna.trial import TrialState
import torch.utils.data
from sklearn.model_selection import train_test_split
from medicaltorch import transforms as mt_transforms
from medicaltorch import datasets as mt_datasets
from medicaltorch import models as mt_models
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

listOfImages = getListOfPPImages(dirName)
listOfMasks = getListOfPPMasks(dirName)
filename_pairs = get_filename_pairs(listOfImages, listOfMasks)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\ndevice:', device, torch.cuda.current_device())

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

# Define preprocessing transformations
preprocessing_transforms = torchvision.transforms.Compose([
    HistogramClipping(),
    RangeNorm(),
    mt_transforms.ToTensor(),
])

# Split patients into two portions: train for cross validation and an independent evaluation portion
train_pat, test_pat = train_test_split(patients, test_size=test_size, shuffle=True, random_state=random_seed)
print(f"Patients in train: {train_pat}, total {len(train_pat)}")
print(f"Patients in test: {test_pat}, total {len(test_pat)}")
blist = []
for entry in listpat:
    if entry in train_pat:
        blist.append(True)
    if entry in test_pat:
        blist.append(False)
train_files = []
test_files = []
for i in range(len(blist)):
    if blist[i] == True:
        train_files.append(filename_pairs[i])
    if blist[i] == False:
        test_files.append(filename_pairs[i])
print(f'Test files:\n {test_files}, total {len(test_files)}')
print(f'Train files:\n {train_files}, total {len(train_files)}')

train_dataset = MRI2DSegmentationDataset(train_files,
                                         preprocess=preprocessing_transforms,
                                         slice_filter_fn=slice_filtering_count0,
                                         )

val_dataset = MRI2DSegmentationDataset(test_files,
                                       preprocess=preprocessing_transforms,
                                       )

# Define dataloaders
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


totalslices = len(train_dataloader.dataset) + len(val_dataloader.dataset)
print(
    f'train/validation/test slice numbers= {len(train_dataloader.dataset)}, split in: '
    f'{len(train_dataloader.dataset) / totalslices}, {len(val_dataloader.dataset) / totalslices}')

# Perform the trials
def objective(trial):
    # Generate the model.
    model = mt_models.Unet()
    model.apply(reset_weights)
    model.to(device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adamax", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    criterion = DiceLoss()

    # Training of the model.
    # Train
    print('start training loop')
    val_losses = []
    train_losses = []
    best_loss = 1
    for epoch in range(num_epochs):
        print(f'Training epoch {epoch + 1}')
        model.train()
        train_loss = 0

        for i, batch in enumerate(tqdm(train_dataloader)):

            images, labels = batch["input"], batch["gt"]
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation of the model.
        print(f'Validating epoch {epoch + 1}')
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                images, labels = batch['input'], batch['gt']
                images, labels = images.to(device), labels.to(device)

                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item()

        mean_train_loss = train_loss / len(train_dataloader)
        print('mean train loss:', mean_train_loss)
        train_losses.append(mean_train_loss)

        mean_val_loss = val_loss / len(val_dataloader)
        print('mean validation loss', mean_val_loss)
        val_losses.append(mean_val_loss)

        scheduler.step(mean_val_loss)

        trial.report(mean_val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save figures with results
    fig1 = optuna.visualization.plot_contour(study, params=["optimizer", "lr"])
    fig1.write_image('/.../SLURMoutput/optim/Contour.png')
    fig2 = optuna.visualization.plot_intermediate_values(study)
    fig2.write_image('/.../SLURMoutput/optim/Intermediate_values.png')
    fig3 = optuna.visualization.plot_optimization_history(study)
    fig3.write_image('/.../SLURMoutput/optim/Optimization_history.png')
    fig4 = optuna.visualization.plot_parallel_coordinate(study, params=["optimizer", "lr"])
    fig4.write_image('/.../SLURMoutput/optim/Parallel_coordinate.png')
    fig5 = optuna.visualization.plot_param_importances(study)
    fig5.write_image('/.../SLURMoutput/optim/Param_importances.png')

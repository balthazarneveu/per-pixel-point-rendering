import torch
from pathlib import Path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAMERA = "camera"
MESH_PATH = Path('__data')
CAMERA_PARAMS_FILE = "camera_params.json"
RGB_VIEW_FILE = "view.png"
X = 0
Y = 1
Z = 2

OUTPUT_FOLDER_NAME = "__output"
DATALOADER = "data_loader"
BATCH_SIZE = "batch_size"
TRAIN, VALIDATION, TEST = "train", "validation", "test"
ID = "id"
NAME = "name"
NB_EPOCHS = "nb_epochs"
ARCHITECTURE = "architecture"
MODEL = "model"
NAME = "name"
N_PARAMS = "n_params"
OPTIMIZER = "optimizer"
LR = "lr"
PARAMS = "parameters"
SCHEDULER_CONFIGURATION = "scheduler_configuration"
SCHEDULER = "scheduler"
REDUCELRONPLATEAU = "ReduceLROnPlateau"
LOSS = "loss"
LOSS_MSE = "mse"
METRICS = "metrics"
METRIC_PSNR = "psnr"
REDUCTION_AVERAGE = "average"
REDUCTION_SKIP = "skip"

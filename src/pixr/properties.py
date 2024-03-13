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

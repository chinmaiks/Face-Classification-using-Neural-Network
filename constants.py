import os

#Directory path
DIR_PATH = "C:/Users/chinm/Documents/Courses/Sem4/CV/Project/3/ckaidab_proj03/"

#Directories inside the directory path
IMAGES_DIR = "data/originalPics/"
FDDB_FILE_PATH = "data/FDDB-folds/"
IMG_FORMAT = ".jpg"

OUTPUT_DIR = "data/output/"
BBOX_FDDB_FOLDS = "bbox-FDDB-folds/"
MODEL_OUTPUTS = "model_output/"
BBOX_FACES_DIR = "faces/"
BBOX_NON_FACES_DIR = "non_faces/"
FACE_DIM = (32, 32)
BATCH_SIZE = 64

#Data selection constants
TRAIN_DATA_TYPE = 'train'
TEST_DATA_TYPE = 'test'
VALID_DATA_TYPE = 'valid'

#LeNet5 Model constants
#Number of training samples: pick equal number of face and non-face data for training
TRAIN_SAMPLES = 4000
TEST_SAMPLES = 100
EPOCHS = 100 #no of EPOCHS in training
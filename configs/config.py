import os


SEED = 156

DATA_PATH = "datasets/"
WOS_PATH = os.path.join(DATA_PATH, "WebOfScience/Meta-data/")
WOS_TAXONOMY = os.path.join(WOS_PATH, "WOS_taxonomy.json")
FORC_PATH = os.path.join(DATA_PATH, "FORC/")
FORC_TAXONOMY = os.path.join(FORC_PATH, "Taxonomy4CL_v1.0.1.json")
LOG_PATH = "logs/"
LOG_NAME = "main"

CHECKPOINT_PATH = os.path.join(LOG_PATH, "model/")

AVERAGES = ["macro", "micro"]

CV_FOLDS = 10

EPOCHS = 10
HIDDEN_DIM = 256
DROPOUT = 0.2
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

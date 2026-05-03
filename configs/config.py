import os
from datetime import datetime


SEED = 156

DATA_PATH = "datasets/"
WOS_PATH = os.path.join(DATA_PATH, "WebOfScience/Meta-data/")
WOS_TAXONOMY = os.path.join(WOS_PATH, "WOS_taxonomy.json")
WOS_ROOT = "root"
FORC_PATH = os.path.join(DATA_PATH, "FORC/")
FORC_TAXONOMY = os.path.join(FORC_PATH, "Taxonomy4CL_v1.0.1.json")
FORC_ROOT = "NLP"
TOY_PATH = os.path.join(DATA_PATH, "toy/")
TOY_TAXONOMY = os.path.join(TOY_PATH, "toy_taxonomy.json")
TOY_ROOT = "NLP"
LOG_PATH = f"logs/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_ANALYSIS = os.path.join(LOG_PATH, "analysis.xlsx")
LOG_LOSS = os.path.join(LOG_PATH, "loss.png")
NPZ = os.path.join(LOG_PATH, "results.npz")

EMB_MODEL = "bert-base-uncased"
EMB_BATCH = 1024
EMB_SIZE = 512

CHECKPOINT_PATH = os.path.join(LOG_PATH, "model/")

AVERAGES = ["macro", "micro"]

CV_FOLDS = 10

EPOCHS = 100
HIDDEN_DIM = 256
DROPOUT = 0.1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LAMBDA = 1
GAMMA = 2
PATIENCE = 5
WEIGHTED = True

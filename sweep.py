"""

"""

import wandb as wb

from main import EnteTransliterator
from atten import EnteTransliteratorAttn
from data import Data

DATA_DIR_PATH = r'data/mal/'
DATA_TRAIN_PATH = r'mal_train.csv'
DATA_TEST_PATH = r'mal_test.csv'
DATA_VALID_PATH = r'mal_valid.csv'

train_data = Data(DATA_DIR_PATH + DATA_TRAIN_PATH)
valid_data = Data(DATA_DIR_PATH + DATA_VALID_PATH)
test_data  = Data(DATA_DIR_PATH + DATA_TEST_PATH)

TYPE = "atten"

def init(sweep_count: int = 1, type_: str = "atten"):

    wandb_project:str = "cs6910-assignment-3"
    wandb_entity:str = "cs22m056"

    global TYPE
    TYPE = type_

    sweep_conf = {

        'method' : 'bayes',
        'metric' : {
        'name' : 'validation_accuracy',
        'goal' : 'maximize'   
        },
        'parameters': {
            'epochs': {
                'values': [10, 20, 30, 40]
            },
            'no_of_encoder_layers': {
                'values': [3, 1, 2]
            },
            'no_of_decoder_layers': {
                'values': [3, 1, 2]
            },
            'hidden_size': {
                'values': [64, 128, 32]
            },
            'cell_type': {
                'values': ['LSTM', 'GRU', 'RNN']
            },
            'learning_rate': {
                'values': [1e-3, 1e-4, 1e-5] 
            },
            'batch_size' : {
                'values':[32, 64, 128]
            },
            'dropout': {
                'values': [0, 0.2, 0.3, 0.5]
            },
            'bidirectional': {
                'values': [True, False]
            },
            'emb': {
                'values': [100, 150, 200]
            },
            'search_method': {
                'values': ['greedy', 'beam']
            },
            'beam_width': {
                'values': [1, 3, 4, 5]
            }
        }
    }

    sweep_id = wb.sweep(sweep_conf, project=wandb_project, entity=wandb_entity)
    wb.agent(sweep_id, sweep, wandb_entity, wandb_project, sweep_count)

def sweep():
    
    wb.init(resume="auto")
    config = wb.config

    name = (
        TYPE + 
        "_cell_" + str(config.cell_type) +
        "_hid_" + str(config.hidden_size) +
        "_bid_" + str(config.emb) +
        "_bat_" + str(config.batch_size)
    )

    wb.run.name = name
    print(name)

    if TYPE == "vanilla":

        transliterator = EnteTransliterator(
            config.cell_type,
            config.epochs,
            config.no_of_encoder_layers,
            config.no_of_decoder_layers,
            config.hidden_size,
            config.learning_rate,
            config.batch_size,
            config.dropout,
            config.bidirectional,
            config.emb,
            config.search_method,
            config.beam_width
        )
    
    if TYPE == "atten":

        transliterator = EnteTransliteratorAttn(
            config.cell_type,
            config.epochs,
            config.hidden_size,
            config.learning_rate,
            config.batch_size,
            config.dropout,
            config.emb,
            config.search_method,
            config.beam_width
        )

    transliterator.train()
    # transliterator.test(test_data)

if __name__ == "__main__":
    init(sweep_count=40, type_ = "atten")

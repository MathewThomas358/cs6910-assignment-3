"""

"""

import sys

from main import EnteTransliterator
from atten import EnteTransliteratorAttn

TYPE = sys.argv[1]

if TYPE == "attn":

    tr = EnteTransliterator(
        cell_type="LSTM",
        epochs=30,
        encoder_layers=2,
        decoder_layers=2,
        hidden_size=128,
        lr=0.001,
        batch_size=128,
        dropout=0,
        bidirectional=True,
        emb=200,
    )

else:

    tr = EnteTransliteratorAttn(
        cell_type="LSTM",
        epochs=40,
        hidden_size=128,
        lr=1e-3,
        batch_size=128,
        dropout=0,
        emb=200
    )

tr.train()
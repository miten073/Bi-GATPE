Hyperparameters of Data Imputation Models

1) iTransformer

Embedding dimension: 64

Number of Transformer layers: 2–3

Number of attention heads: 4

Feed-forward hidden dimension: 128

Dropout: 0.1

Optimizer: RMSprop

Loss function: MAE


2) BRITS

RNN type: LSTM

Hidden size: 64

Number of layers: 2

Bidirectional: True

Dropout: 0.1

Optimizer: RMSprop

Loss function: MAE


3) FreTS

Frequency-domain hidden dimension: 64

Number of MLP layers: 2–3

Activation: GELU

Dropout: 0.1

Optimizer: RMSprop

Loss function: MAE


4) TimesNet

Embedding dimension: 64

Number of TimesBlocks: 2

Inception kernel number: 3–5

Hidden dimension: 128

Dropout: 0.1

Optimizer: RMSprop

Loss function: MAE


5) SAITS

DMSA blocks: 2

Attention heads: 4

Model dimension: 64

FFN hidden dimension: 128

Dropout: 0.1

Optimizer: RMSprop

Loss function: MAE


6) PatchTST

Patch length: 8–16

Stride: 4

Embedding dimension: 64

Transformer layers: 2

Attention heads: 4

Dropout: 0.1

Optimizer: RMSprop

Loss function: MAE


7) Mamba

State dimension: 64

Number of layers: 2

Expansion factor: 2

Dropout: 0.1

Optimizer: RMSprop

Loss function: MAE

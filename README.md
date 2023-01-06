# SmartRep project files

`SmartRep/`: implement of our approach

`seq2seq_attn/`: implement of sequence-to-sequence model with attention mechanism

`main.py`: begin training

`validationn.py`: begin testing

`utils.py`: some useful files of implement 

`dataset/`: dataset for training and testing

## Usage

environment

```shell
pip install -r requirements.txt
```

train

```shell
python main.py [method] [dataset] [start]
# for example:
python main.py SmartRep total -1  # set "start" as -1 to train a new model
```

test

```shell
python validationn.py [method] [dataset] [path]
```


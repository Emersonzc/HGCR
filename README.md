![Python >=3.5](https://img.shields.io/badge/Python->=3.6-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-yellow.svg)

# Hard-sample guided cluster refinement for Unsupervised Person Re-Identification

The *official* repository for [Hard-sample guided cluster refinement for Unsupervised Person Re-Identification].

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets Market-1501,PersonX, and DukeMTMC-reID.
Then unzip them under the directory like

```
hgcr/examples/data
├── market1501
│   └── Market-1501-v15.09.15
├── personx
│   └── PersonX
├── dukemtmcreid
    └── DukeMTMC-reID
```

## Training

We utilize 4 GTX-2080 GPUs for training. For more parameter configuration, please check **`run_code.sh`**.

**examples:**

Market-1501:

1. Using DBSCAN:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/hgcr_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
```

DukeMTMC-reID:

1. Using DBSCAN:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/hgcr_usl.py -b 256 -a resnet50 -d dukemtmcreid --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16
```

## Evaluation

We utilize 1 GTX-2080 GPU for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets, and `--height 224 --width 224` for vehicle datasets;


To evaluate the model, run:
```shell
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
  -d $DATASET --resume $PATH
```

**Some examples:**
```shell
### Market-1501 ###
CUDA_VISIBLE_DEVICES=0 \
python examples/test.py \
  -d market1501 --resume logs/hgcr_usl/market_resnet50/model_best.pth.tar
```



# Acknowledgements

Thanks to Zuozhuo Dai for opening source of his excellent works  [cluter contrast](https://github.com/alibaba/cluster-contrast-reid). 
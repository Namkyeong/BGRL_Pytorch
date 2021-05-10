# BGRL_Pytorch
Implementation of Bootstrapped Representation Learning on Graphs.

A PyTorch implementation of "<a href="https://arxiv.org/pdf/2102.06514.pdf">Bootstrapped Representation Learning on Graphs</a>" paper, accepted in ICLR 2021 Workshop

## Hyperparameters for training BGRL
Following Options can be passed to `train.py`

`--layers` or `-l`  
one or more integer values specifying  the number of units for each GNN layer.  
Default is 512 128  
`--layers 512 128`  

`--aug_params` or `-p`  
four float values specifying the hyperparameters for graph augmentation (p_f1, p_f2, p_e1, p_e2).  
Default is 0.2 0.1 0.2 0.3  
`--layers 0.2 0.1 0.2 0.3`  


## Codes borrowed from
Codes are borrowed from BYOL and SelfGNN


| name        | Implementation Code | Paper   |
| ----------- | ------------------- | ------- | 
| `Bootstrap Your Own Latent`| <a href="https://github.com/lucidrains/byol-pytorch">Implementation</a>| <a href="https://arxiv.org/pdf/2006.07733.pdf">paper</a>|
| `SelfGNN`| <a href="https://github.com/zekarias-tilahun/SelfGNN">Implementation</a>| <a href="https://arxiv.org/pdf/2103.14958.pdf">paper</a>|

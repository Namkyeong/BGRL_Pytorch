rm -rf data
python train.py --layers 256 --device 0 --epochs 10000 --aug_params 0.3 0.4 0.3 0.2
python train.py --layers 512 256 --device 0 --epochs 10000 --aug_params 0.3 0.4 0.3 0.2

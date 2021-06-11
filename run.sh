rm -rf data
python train.py --layers 256 --device 2 --epochs 2000 --aug_params 0.3 0.4 0.3 0.2
rm -rf data
python train.py --layers 256 --device 2 --epochs 2000 --aug_params 0.5 0.4 0.1 0.2
rm -rf data
python train.py --layers 256 --device 2 --epochs 2000 --aug_params 0.1 0.2 0.5 0.4
rm -rf data
python train.py --layers 256 --device 2 --epochs 2000 --aug_params 0.2 0.1 0.1 0.2
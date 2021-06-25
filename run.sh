rm -rf data
python train.py --device 0 --epochs 10000 --name WikiCS --aug_params 0.2 0.1 0.2 0.3 --lr 0.0005 --layers 512 256 --pred_hid 512
python train.py --device 0 --epochs 10000 --name computers --aug_params 0.2 0.1 0.5 0.4 --lr 0.0005 --layers 256 128 --pred_hid 512
python train.py --device 0 --epochs 10000 --name photo --aug_params 0.1 0.2 0.4 0.1 --lr 0.0001 --layers 512 256 --pred_hid 512
python train.py --device 0 --epochs 10000 --name cs --aug_params 0.3 0.4 0.3 0.2 --lr 0.00001 --layers 512 256 --pred_hid 512
python train.py --device 0 --epochs 10000 --name physics --aug_params 0.1 0.4 0.4 0.1 --lr 0.00001 --layers 256 128 --pred_hid 512
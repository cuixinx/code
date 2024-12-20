dataset='office-home'
output_dir=res/ckps/source/
python train_source.py --output $output_dir --dset ${dataset} --trte val --net resnet101 --lr 1e-3 --max_epoch 10 --s 0;

#CUDA_VISIBLE_DEVICES=1 python train_source.py --output $output_dir --dset ${dataset} --trte val --net resnet50 --lr 1e-3 --max_epoch 60 --s 0&
CUDA_VISIBLE_DEVICES=2 python train_source.py --output $output_dir --dset ${dataset} --trte val --net resnet50 --lr 1e-3 --max_epoch 60 --s 1&
CUDA_VISIBLE_DEVICES=3 python train_source.py --output $output_dir --dset ${dataset} --trte val --net resnet50 --lr 1e-3 --max_epoch 60 --s 2&
CUDA_VISIBLE_DEVICES=4 python train_source.py --output $output_dir --dset ${dataset} --trte val --net resnet50 --lr 1e-3 --max_epoch 60 --s 3;


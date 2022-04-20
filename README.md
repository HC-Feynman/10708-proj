
This is the implementation for 10708 course project "Oracle Bone Character
Generation Using Image-to-Image Translation" by Hui Chen, Zhihan Lu, Jiaqi Zeng. It is modified from the [code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by the authors of  [Paper](https://arxiv.org/pdf/1611.07004.pdf).

To run the baseline Pix2pix model:
```
python train.py --dataroot ./datasets/oracle-6h-4k --name baseline --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 70 --print_freq 700  --save_latest_freq 14000  --n_epochs  100   --n_epochs_decay  0

python test.py --dataroot ./datasets/oracle-6h-4k --name baseline --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 246 
```

To run Pix2pix + semi-supervised learning:
```
python train.py --dataroot ./datasets/oracle-6h-4k --name semisup --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 70 --print_freq 700  --save_latest_freq 14000  --n_epochs  100   --n_epochs_decay  0 --semi_sup 1

python test.py --dataroot ./datasets/oracle-6h-4k --name semisup --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 246 --semi_sup 1
```

For the following two methods, a pretrained VAE model is required.
Download the pretrained model into ./ from [pretrained_G.pth](https://drive.google.com/drive/folders/1a-PK2vqxkbQwQy6KULkQvqNME6qJoebX?usp=sharing) (CMU account is required) or you can pretrain it by yourself
TBA


To run Pix2pix + VAE:
```
python train.py --dataroot ./datasets/oracle-6h-4k --name proposed --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 100 --print_freq 1000  --save_latest_freq 10000  --n_epochs  0   --n_epochs_decay  0

cp ./pretrained_G.pth checkpoints/proposed/latest_net_G.pth 

python train.py --dataroot ./datasets/oracle-6h-4k --name baseline --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 70 --print_freq 700  --save_latest_freq 14000  --n_epochs  100   --n_epochs_decay  0

python test.py --dataroot ./datasets/oracle-6h-4k --name baseline --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 246 
```

To run our final method (Pix2pix + semi-supervised learning + VAE):
```
python train.py --dataroot ./datasets/oracle-6h-4k --name proposed --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 100 --print_freq 1000  --save_latest_freq 10000  --n_epochs  0   --n_epochs_decay  0

cp ./pretrained_G.pth checkpoints/proposed/latest_net_G.pth 

python train.py --dataroot ./datasets/oracle-6h-4k --name semisup --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 70 --print_freq 700  --save_latest_freq 14000  --n_epochs  100   --n_epochs_decay  0 --semi_sup 1

python test.py --dataroot ./datasets/oracle-6h-4k --name semisup --model pix2pix --direction BtoA --input_nc 1 --output_nc 1 --num_test 246 --semi_sup 1
```
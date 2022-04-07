# How to use vae-pretrained model

Do the following steps:

- Modify train.py to enable saving the model before the actual training process. Specifically, add

  ```python
  model.save_networks('latest') # before training
  ```

  before

  ```python
  for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
  ```

- Train a new model for zero epoch.

  ```bash
  python train.py --dataroot ./datasets/oracle-6h-4k --name new_model_name --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 100 --print_freq 1000  --save_latest_freq 10000  --n_epochs  0   --n_epochs_decay  0
  
  ```

- Replace checkpoints/new_model_name/latest_net_G.pth with the vae-pretrained model [here](https://drive.google.com/drive/folders/1a-PK2vqxkbQwQy6KULkQvqNME6qJoebX?usp=sharing).

- Do the full training.

  ```bash
  python train.py --dataroot ./datasets/oracle-6h-4k --name new_model_name --model pix2pix --input_nc 1 --output_nc 1 --direction BtoA --display_id 0 --batch_size 100 --print_freq 1000  --save_latest_freq 10000  --n_epochs  100   --n_epochs_decay  100 --continue_train 
  
  ```

  
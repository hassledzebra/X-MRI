import datetime
from pathlib import Path
from argparse import ArgumentParser
from typing import List

from ssl_transform_kspace_2 import SslTransform
import pytorch_lightning as pl
from fastmri.data import transforms as fastmri_transforms

from fastmri.pl_modules import FastMriDataModule

import fastmri
from fastmri.data import subsample
import numpy as np
import torch.optim
from torch.utils.data import DataLoader
import os

# Tqdm progress bar
from tqdm import tqdm_notebook, tqdm

import sys
from matplotlib import pyplot as plt



from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# Define environment variables for BART and OpenMP

os.environ['TOOLBOX_PATH'] = "/content/bart"

os.environ['OMP_NUM_THREADS']="4"

# Add the BARTs toolbox to the PATH variable

os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python")

from bart import bart


def handle_args():
    parser = ArgumentParser()

    #num_gpus = 0
    #backend = "ddp_cpu"
    #batch_size = 1 if backend == "ddp" else num_gpus
    #batch_size = 2

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        dest='mode',
        help="Operation mode"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        type=str,
        dest='device',
        help="Device type",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        dest='data_path',
        help="Path to data",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        dest='checkpoint_path',
        help="When train, dir path for saving model checkpoints; when test, either director (from which to load newest"
             " checkpoint) or specific checkpoint file to load",
    )
    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.2],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerator",
        dest='accelerator',
        default='ddp',
        help="What distributed version to use",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument("--non_deterministic", action='store_false', default=True, dest='deterministic')
    parser.add_argument("--replace_sampler_ddp", action='store_true', default=False, dest='replace_sampler_ddp',
                        help="Replace sampler ddp")
    parser.add_argument("--seed", default=42, dest='seed', help="Seed for all the random generators")
    parser.add_argument("--num_gpus", default=1, help="The number of available GPUs (when device is 'cuda'")

    return parser.parse_args()


def get_sorted_checkpoint_files(checkpoint_dir: Path) -> List[Path]:
    files = list(checkpoint_dir.glob('*.pt'))
    files.sort()
    return files


def save_checkpoint(optimizer, epoch, model: torch.nn.Module, checkpoint_dir: Path, loss, limit=None):
    filename = 'ssl_sd_checkpoint_{}.pt'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss
          },checkpoint_dir.joinpath(filename))
    else:
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss
          },checkpoint_dir.joinpath(filename))
        files = get_sorted_checkpoint_files(checkpoint_dir)
        if limit and len(files) > limit:
            files[0].unlink()


def load_from_checkpoint(optimizer, model: torch.nn.Module, checkpoint_dir: Path, specific_file: str = None, set_eval=True):
    if specific_file is None:
        files = get_sorted_checkpoint_files(checkpoint_dir)
        file_path = files[-1]
    else:
        file_path = checkpoint_dir.joinpath(specific_file)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'previous loss:  %.7f' % loss.item())
    if set_eval:
        model.eval()
    return optimizer, model, epoch


def calc_ssl_loss(u, v):
    abs_u_minus_v = torch.sum(torch.abs(u - v)) #added torch sum
    abs_u = torch.sum(torch.abs(u)) # added torch sum
    term_1 = torch.pow(abs_u_minus_v, 2) / torch.pow(abs_u, 2)
    term_2 = abs_u_minus_v / abs_u
    return (term_1 + term_2) # divide by image size or not


def choose_loss_split(volume, ratio=0.5):
    # TODO: come back and implement overlap
    # volume format: (image, target, mean, std, fname, slice_num, max_value)
    volume = volume[0].cuda() #make sure on the same device the first is train and second is target. only using train now
    #volume = volume.permute(0,3,1,2)
    # volume.shape: N, 320, 320, 2
    arange = np.arange(volume.shape[2])
    rng = np.random.default_rng() # corrected the random generator
    theta_indices = rng.choice(arange, size=int(volume.shape[2] * ratio), replace=False)
    lambda_indices = arange[np.isin(arange, theta_indices, invert=True)]

    volume_theta_view, volume_lambda_view = torch.zeros_like(volume),torch.zeros_like(volume)
    #volume_theta_view[:,theta_indices,:] = volume[:,theta_indices,:]
    volume_theta_view[:,:,theta_indices,:] = volume[:,:,theta_indices,:]
    #volume_lambda_view[:,lambda_indices,:] = volume[:,lambda_indices,:]
    volume_lambda_view[:,:,lambda_indices,:] = volume[:,:,lambda_indices,:]
    return volume_theta_view, volume_lambda_view,lambda_indices


def run_training_for_volume(volume, model: torch.nn.Module, optimizer, cs):
   
    #cs = 0 # whether to use compressed sensing
    
    if cs:
      print(volume[2].shape)
      temp_volume = volume[2] #shape: N, 320, 320, 2 using the target k-space fully sampled
      with suppress_stdout():
        temp_volume = temp_volume.cpu()
      temp_volume = fastmri.tensor_to_complex_np(temp_volume) # shape N, 320, 320
      sens = np.zeros_like(temp_volume)
      for i in range(temp_volume.shape[0]):
        with suppress_stdout():
          sens[i] = bart(1, 'ecalib -d0 -m1', temp_volume[i]).squeeze()
        


    volume_theta_view, volume_lambda_view, lamda_indices = choose_loss_split(volume)
    
    
    volume_lambda_view, _, _ = fastmri_transforms.normalize_instance(volume_lambda_view, eps=1e-11)
    
    volume_lambda_view = volume_lambda_view.clamp(-6, 6)
    if cs:
      v_ifft = fastmri.ifft2c(volume_theta_view)
      v_ifft, _, __builtins__ = fastmri_transforms.normalize_instance(v_ifft, eps=1e-11)
      v_ifft = v_ifft.clamp(-6, 6)
      volume_theta_view = volume_theta_view.cpu()
      volume_theta_view = fastmri.tensor_to_complex_np(volume_theta_view) # shape N, 320, 320
      temp = np.zeros_like(volume_theta_view)
      for i in range(volume_theta_view.shape[0]):
        with suppress_stdout():
          pred = bart(1, f"pics -d0 -S -R T:7:0:0.1 -i 10", volume_theta_view[i], sens[i])
        temp[i] = pred
      #volume_theta_view = fastmri_transforms.to_tensor(temp).cuda()
      v_bart = fastmri_transforms.to_tensor(temp).cuda()
      v_bart, _, _ = fastmri_transforms.normalize_instance(v_bart, eps=1e-11)
      v_bart = v_bart.clamp(-6, 6)
      volume_theta_view = torch.cat((v_ifft, v_bart),-1) # shape: N, 320, 320, 4
    else:
      volume_theta_view = fastmri.ifft2c(volume_theta_view)
      volume_theta_view ,_,_=fastmri_transforms.normalize_instance(volume_theta_view, eps=1e-11)
      volume_theta_view = volume_theta_view.clamp(-6,6)
    
    
    volume_theta_view = volume_theta_view.permute(0,3,1,2) # shape: N, 2, 320, 320
    prediction = model(volume_theta_view)
    
    prediction = prediction.permute(0,2,3,1) #shape: N, 320, 320, 2
    prediction = fastmri.fft2c(prediction)
    prediction, _, _ = fastmri_transforms.normalize_instance(prediction, eps=1e-11)
    prediction = prediction.clamp(-6, 6)
    pred_lamda = torch.zeros_like(prediction)
    #pred_lamda[:,:,lamda_indices,:] = prediction[:,:,lamda_indices,:] # only comparing the lamda view
    pred_lamda[:,:,lamda_indices,:] = prediction[:,:,lamda_indices,:] # only comparing the lamda view
    
    loss = calc_ssl_loss(u=volume_lambda_view, v=pred_lamda)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def run_training_eval(cs, model: torch.nn.Module, checkpoint_dir: Path, data_module, epochs=100):
    training_data = data_module.train_dataloader()
    val_data = data_module.val_dataloader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    EPOCH = 0

    dir = os.listdir(checkpoint_dir)
    if len(dir) != 0:
      print('training from check point: ')
      optimizer, model, EPOCH = load_from_checkpoint(optimizer, model, checkpoint_dir)
      EPOCH += 1
    
    for e in range(EPOCH, epochs):
        print("-----------------------------------")
        print("Epoch %d" % (e+1))
        print("-----------------------------------")
        total_loss = 0
        
        progress_bar = tqdm(training_data)
        
        model.train()
        for idx, volume in enumerate(progress_bar):
            
            loss = run_training_for_volume(volume, model, optimizer,cs)
            total_loss += loss
            progress_bar.set_description_str(
              "Batch: %d, Loss: %.7f" % ((idx + 1), loss.item()))
        #scheduler.step(loss)
        save_checkpoint(optimizer, e, model, checkpoint_dir, total_loss/len(training_data))
        print(f"loss: {total_loss/len(training_data):>7f}  [{e+1:>5d}/{epochs:>5d}]")
        
        val = False
        if val == True:
          total_loss = 0
          progress_bar = tqdm(val_data)
          model.eval()
         
          for idx, volume in enumerate(progress_bar):
            src = volume[0].unsqueeze(1).cuda()
            pred = model(src)
            target = volume[1].unsqueeze(1).cuda()
            loss = calc_ssl_loss(pred, target)
            total_loss += loss
            progress_bar.set_description_str(
              "Batch: %d, validation Loss: %.7f" % ((idx + 1), loss.item()))
          
        print(f"validation loss: {total_loss/len(training_data):>7f}  [{e+1:>5d}/{epochs:>5d}]")


def run_pretrained_inference(checkpoint_source: Path):
    # TODO: source may be directory (newest file) or actual file to load
    # TODO: implement
    raise RuntimeError("Testing mode not currently supported")


def main():
    args = handle_args()
    pl.seed_everything(args.seed)

    # creates k-space mask for transforming
    mask = subsample.create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    train_transform = SslTransform(mask_func=mask, use_seed=False)
    val_transform = SslTransform(mask_func=mask)
    test_transform = SslTransform()

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge='singlecoil',
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split='test',
        # TODO: this in particular might need to be changed
        test_path=None,
        sample_rate=1,
        batch_size=2,
        num_workers=2,
        #distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cs = 1
    if cs:
      from self_supervised_2 import MriSelfSupervised
    else:
      from self_supervised import MriSelfSupervised
    checkpoint_path = args.checkpoint_path
    if args.mode == "train":
        if checkpoint_path.exists() and not checkpoint_path.is_dir():
            raise RuntimeError("Existing, non-directory path {} given for checkpoint directory".format(checkpoint_path))
        
        model = MriSelfSupervised()
        #print('model_params',model.parameters)
        model.to(device)
        run_training_eval(cs, model=model, checkpoint_dir=checkpoint_path, data_module=data_module)
    elif args.mode == "test":
        if not checkpoint_path.exists():
            raise RuntimeError("Non-existing checkpoint file/directory path {}".format(checkpoint_path))
        run_pretrained_inference(checkpoint_source=checkpoint_path)
    else:
        raise RuntimeError("Unsupported mode '{}'".format(args.mode))


if __name__ == "__main__":
    main()
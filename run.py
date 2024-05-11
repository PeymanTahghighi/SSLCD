
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from monai.losses.dice import DiceLoss
from monai.metrics.confusion_matrix import get_confusion_matrix
from utility import BounraryLoss
from VNet import VNet, VNetCrossAttention
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.unetr import UNETR
import argparse
from torchvision.utils import save_image, make_grid
from data_utils import cache_dataset_WHU, get_loader_WHU



def save_examples_miccai(model, loader,):
    """save examples for segmentation of new lesions from MICCAI-21 dataset

    Parameters
    ----------
    model : nn.Module
        model to use for predictions
    
    loader : DataLoader
        data loader to iterate through
    """
    if os.path.exists('samples') is False:
        os.makedirs('samples');
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        counter = 0;
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap, lbl, brainmask = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), batch[3], batch[4].to(args.device);
            if torch.sum(heatmap).item() > 0:
                hm1 = model(mri, mri_noisy);
                hm2 = model(mri_noisy, mri);
                heatmap = heatmap.detach().cpu().numpy();
                pred_lbl_1 = torch.sigmoid(hm1)>0.5;
                pred_lbl_2 = torch.sigmoid(hm2)>0.5;
                pred = (pred_lbl_1 * pred_lbl_2*brainmask).detach().cpu().numpy();
                hm1 = hm1.detach().cpu().numpy();
                hm2 = hm2.detach().cpu().numpy();
                mri = mri.detach().cpu().numpy();
                mri_noisy = mri_noisy.detach().cpu().numpy();

                for j in range(2):
                    pos_cords = np.where(heatmap[0] >0);
                    if len(pos_cords[0]) != 0:
                        r = np.random.randint(0,len(pos_cords[0]));
                        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
                    else:
                        center = [hm1.shape[2]//2, hm1.shape[3]//2, hm1.shape[4]//2]
                    fig, ax = plt.subplots(2,6);
                    ax[0][0].imshow(pred[0,0,center[0], :, :], cmap='hot');
                    ax[0][1].imshow(pred[0,0,:,center[1], :], cmap='hot');
                    ax[0][2].imshow(pred[0,0, :, :,center[2]], cmap='hot');

                    ax[0][3].imshow(heatmap[0,0,center[0], :, :]);
                    ax[0][4].imshow(heatmap[0,0,:,center[1], :]);
                    ax[0][5].imshow(heatmap[0,0, :, :,center[2]]);

                    ax[1][0].imshow(mri[0,0,center[0], :, :], cmap='gray');
                    ax[1][1].imshow(mri[0,0,:,center[1], :], cmap='gray');
                    ax[1][2].imshow(mri[0,0, :, :,center[2]], cmap='gray');

                    ax[1][3].imshow(mri_noisy[0,0,center[0], :, :], cmap='gray');
                    ax[1][4].imshow(mri_noisy[0,0,:,center[1], :], cmap='gray');
                    ax[1][5].imshow(mri_noisy[0,0, :, :,center[2]], cmap='gray');

                    fig.savefig(os.path.join('samples',f'sample_{epoch}_{counter + idx*args.batch_size}_{j}.png'));
                    plt.close("all");

                counter += 1;
                if counter >=5:
                    break;

def save_examples(model, loader, path):
    """save examples for self-supervised pretrained model, it will predict all the changes from one MRI scan to the other

    Parameters
    ----------
    model : nn.Module
        model to use for predictions
    
    loader : DataLoader
        data loader to iterate through
    """
    if os.path.exists(os.path.join(path, 'samples')) is False:
        os.makedirs(os.path.join(path, 'samples'));
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        counter = 0;
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap = batch[0].to(args.device).unsqueeze(dim=1), batch[1].to(args.device).unsqueeze(dim=1), batch[2].to(args.device).unsqueeze(dim=1)
            volume_batch1 = torch.cat([mri, mri_noisy], dim=1)
            volume_batch2 = torch.cat([mri_noisy, mri], dim=1)
            hm1 = model(volume_batch1);
            hm2 = model(volume_batch2);            
            mri_recon = (mri_noisy+hm2);
            mri_noisy_recon = (mri+hm1);

            heatmap = heatmap.detach().cpu().numpy();
            hm1 = hm1.detach().cpu().numpy();
            hm2 = hm2.detach().cpu().numpy();
            mri_recon = mri_recon.detach().cpu().numpy();
            mri_noisy_recon = mri_noisy_recon.detach().cpu().numpy();
            mri = mri.detach().cpu().numpy();
            mri_noisy = mri_noisy.detach().cpu().numpy();
            for j in range(2):
                pos_cords = np.where(heatmap[0] >0);
                if len(pos_cords[0]) != 0:
                    r = np.random.randint(0,len(pos_cords[0]));
                    center = [pos_cords[0][r], pos_cords[1][r],pos_cords[2][r]]
                else:
                    center = [hm1.shape[2]//2, hm1.shape[3]//2, hm1.shape[4]//2]
                fig, ax = plt.subplots(3,6);
                ax[0][0].imshow(hm1[0,0,center[0], :, :], cmap='hot');
                ax[0][1].imshow(hm1[0,0,:,center[1], :], cmap='hot');
                ax[0][2].imshow(hm1[0,0, :, :,center[2]], cmap='hot');

                ax[0][3].imshow(hm2[0,0,center[0], :, :], cmap='hot');
                ax[0][4].imshow(hm2[0,0,:,center[1], :], cmap='hot');
                ax[0][5].imshow(hm2[0,0, :, :,center[2]], cmap='hot');

                ax[1][0].imshow(mri[0,0,center[0], :, :], cmap='gray');
                ax[1][1].imshow(mri[0,0,:,center[1], :], cmap='gray');
                ax[1][2].imshow(mri[0,0, :, :,center[2]], cmap='gray');

                ax[1][3].imshow(mri_noisy[0,0,center[0], :, :], cmap='gray');
                ax[1][4].imshow(mri_noisy[0,0,:,center[1], :], cmap='gray');
                ax[1][5].imshow(mri_noisy[0,0, :, :,center[2]], cmap='gray');

                ax[2][0].imshow(mri_noisy_recon[0,0,center[0], :, :], cmap='gray');
                ax[2][1].imshow(mri_noisy_recon[0,0,:,center[1], :], cmap='gray');
                ax[2][2].imshow(mri_noisy_recon[0,0, :, :,center[2]], cmap='gray');

                ax[2][3].imshow(mri_recon[0,0,center[0], :, :], cmap='gray');
                ax[2][4].imshow(mri_recon[0,0,:,center[1], :], cmap='gray');
                ax[2][5].imshow(mri_recon[0,0, :, :,center[2]], cmap='gray');
                fig.savefig(os.path.join(path, 'samples',f'sample_{epoch}_{counter + idx*args.batch_size}_{j}.png'));
                plt.close("all");
            
            counter += 1;
            #only save 5 samples
            if counter >=5:
                break;

def train_step(args, model, train_loader, optimizer, scalar):
    """one epoch of new lesion segmentation model training

    Parameters
    ----------
    model : nn.Module
        model to use for predictions
    
    train_loader : DataLoader
        data loader to iterate through
    
    optimizer : torch.nn.optim
        optimizer to minimze loss function

    scaler: torch.amp.cuda.scaler
        for mixed precision training
    """
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'IoU'));
    pbar = tqdm(enumerate(train_loader), total= len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
    epoch_IoU = [];
    curr_step = 0;
    curr_iou = 0;
    for batch_idx, (batch) in pbar:
        before, after, label, dt = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), batch[3].to(args.device);

        volume_batch1 = torch.cat([before, after], dim=1)
        volume_batch2 = torch.cat([after, before], dim=1)

        curr_loss = 0;
        with torch.cuda.amp.autocast_mode.autocast():

            
            hm1 = model(volume_batch1);
            hm2 = model(volume_batch2);

            lhf1 = DiceLoss(sigmoid=True)(hm1, label);
            lhf2 = DiceLoss(sigmoid=True)(hm2, label);

            lhb1 = BounraryLoss(sigmoid=True)(hm1, label)*args.bl_multiplier;
            lhb2 = BounraryLoss(sigmoid=True)(hm2, label)*args.bl_multiplier;
            lhh = DiceLoss()(torch.sigmoid(hm1), torch.sigmoid(hm2));
            loss = (lhf1 + lhf2 + lhb1 + lhb2 + lhh)/ args.virtual_batch_size;

            scalar.scale(loss).backward();
            curr_loss += loss.item();
            curr_step+=1;
            curr_iou += (1-(DiceLoss(sigmoid=True)(hm1, label)).item());

            if (curr_step) % args.virtual_batch_size == 0:
                scalar.step(optimizer);
                scalar.update();
                
                model.zero_grad(set_to_none = True);
                epoch_loss.append(curr_loss);
                epoch_IoU.append(curr_iou);
                curr_loss = 0;
                curr_step = 0;
                curr_iou = 0;

            pbar.set_description(('%10s' + '%10.4g'*2)%(epoch, np.mean(epoch_loss), np.mean(epoch_IoU)));

    return np.mean(epoch_loss);

def train_miccai_pretrain(args, model, train_loader, optimizer, scalar):
    """one epoch of self-supervised pretraining

    Parameters
    ----------
    model : nn.Module
        model to use for predictions
    
    train_loader : DataLoader
        data loader to iterate through
    
    optimizer : torch.nn.optim
        optimizer to minimze loss function
        
    scaler: torch.amp.cuda.scaler
        for mixed precision training
    """
    print(('\n' + '%10s'*3) %('Epoch', 'Loss', 'IoU'));
    pbar = tqdm(enumerate(train_loader), total= len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
    epoch_IoU = [];
    curr_step = 0;
    curr_iou = 0;
    for batch_idx, (batch) in pbar:
        mri, mri_noisy, heatmap = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
       
        curr_loss = 0;

        volume_batch1 = torch.cat([mri, mri_noisy], dim=1)
        volume_batch2 = torch.cat([mri_noisy, mri], dim=1)

        with torch.cuda.amp.autocast_mode.autocast():
            hm1 = model(volume_batch1);
            hm2 = model(volume_batch2);

            mri *= (1-heatmap);
            mri_noisy *= (1-heatmap);


            lih1 = F.mse_loss((mri+hm1), mri_noisy, reduce=False);
            lih2 = F.mse_loss((mri_noisy+hm2), mri, reduce=False);
            if args.loss_type == 0:
                lhh = 0;
            if args.loss_type == 1:
                lhh = F.l1_loss((hm1+hm2), torch.zeros_like(hm1), reduce=False)*args.regularize_weight;
            elif args.loss_type == 2:
                lhh = F.l1_loss(hm1, -hm2, reduce=False)*args.regularize_weight;
            elif args.loss_type == 3:
                lhh = F.mse_loss(hm1, -hm2, reduce=False)*args.regularize_weight;
            
            # lh1 = F.l1_loss((hm1)*curr_heatmap, torch.zeros_like(hm1)) * args.mask_loss_multiplier;
            # lh2 = F.l1_loss((hm2)*curr_heatmap, torch.zeros_like(hm2)) * args.mask_loss_multiplier;

            loss = (lih1 + lih2 + lhh)/ args.virtual_batch_size;
            loss = loss.mean();

            scalar.scale(loss).backward();
            curr_loss += loss.item();
            curr_step+=1;


            if (curr_step) % args.virtual_batch_size == 0:
                scalar.step(optimizer);
                scalar.update();
                
                model.zero_grad(set_to_none = True);
                epoch_loss.append(curr_loss);
                epoch_IoU.append(curr_iou);
                curr_loss = 0;
                curr_step = 0;
                curr_iou = 0;
            pbar.set_description(('%10s' + '%10.6f'*1)%(epoch, np.mean(epoch_loss)));

    return np.mean(epoch_loss);

def valid_step(args, model, loader):
    """one epoch of validation

    Parameters
    ----------
    model : nn.Module
        model to use for predictions
    
    loader : DataLoader
        data loader to iterate through
    
    """
    print(('\n' + '%10s'*3) %('Epoch', 'IoU', 'F1'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
    epoch_iou = [];
    epoch_f1 = [];
    with torch.no_grad():
        for idx, (batch) in pbar:
            before, after, label = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)

            volume_batch1 = torch.cat([before, after], dim=1)
            volume_batch2 = torch.cat([after, before], dim=1)

            hm1 = model(volume_batch1);
            hm2 = model(volume_batch2);
            
            pred_lbl_1 = torch.sigmoid(hm1)>0.5;
            pred_lbl_2 = torch.sigmoid(hm2)>0.5;
            pred = pred_lbl_1 * pred_lbl_2;
            curr_iou = (1-(DiceLoss(sigmoid=True)(hm1, label)).item());
            epoch_iou.append(curr_iou);

            cm = get_confusion_matrix(label, pred).squeeze();
            prec = (cm[0])/(cm[0] + cm[1] + 1e-6);
            rec = (cm[0]) /(cm[0] + cm[-1] + 1e-6);
            f1 = (2*prec*rec) / (prec + rec + 1e-6)
            epoch_f1.append(f1.item());
            
            pbar.set_description(('%10s' + '%10.4g'*2) %(epoch, np.mean(epoch_iou), np.mean(epoch_f1)));

    return np.mean(epoch_iou);

def valid_pretrain_miccai(args, model, loader):
    """one epoch of validating self-supervised pretraining model

    Parameters
    ----------
    model : nn.Module
        model to use for predictions
    
    loader : DataLoader
        data loader to iterate through
    
    dataset : Dataset
        used to append patches together and calculate the final results
    """
    print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
    pbar = tqdm(enumerate(loader), total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    epoch_loss = [];
    with torch.no_grad():
        for idx, (batch) in pbar:
            mri, mri_noisy, heatmap = batch[0].to(args.device).unsqueeze(dim=1), batch[1].to(args.device).unsqueeze(dim=1), batch[2].to(args.device).unsqueeze(dim=1)
            volume_batch1 = torch.cat([mri, mri_noisy], dim=1)
            volume_batch2 = torch.cat([mri_noisy, mri], dim=1)
           
            hm1 = model(volume_batch1);
            hm2 = model(volume_batch2);

            mri *= (1-heatmap);
            mri_noisy *= (1-heatmap);

            lih1 = F.mse_loss((mri+hm1), mri_noisy, reduce=False);
            lih2 = F.mse_loss((mri_noisy+hm2), mri, reduce=False);

            if args.loss_type == 0:
                lhh = 0;
            if args.loss_type == 1:
                lhh = F.l1_loss((hm1+hm2), torch.zeros_like(hm1), reduce=False)*args.regularize_weight;
            elif args.loss_type == 2:
                lhh = F.l1_loss(hm1, -hm2, reduce=False)*args.regularize_weight;
            elif args.loss_type == 3:
                lhh = F.mse_loss(hm1, -hm2, reduce=False)*args.regularize_weight;
            
            # lh1 = F.l1_loss((hm1)*curr_heatmap, torch.zeros_like(hm1)) * args.mask_loss_multiplier;
            # lh2 = F.l1_loss((hm2)*curr_heatmap, torch.zeros_like(hm2)) * args.mask_loss_multiplier;

            loss = (lih1 + lih2 + lhh)/ args.virtual_batch_size;
            loss = loss.mean();


            epoch_loss.append(loss.item());
            pbar.set_description(('%10s' + '%10.4g')%(epoch, np.mean(epoch_loss)));

    return np.mean(epoch_loss);

def log_hyperparameters(args):
    if os.path.exists('exp') is False:
        os.mkdir('exp');
    
    print('\n\n*********Hyperparameters*********\n');
    print("{}".format(args).replace(', ', ',\n'));

    if args.resume is False:
        #find a path for summary writer
        exp_id = np.random.randint(0,9999999);
        path_to_sum_wr = os.path.join('exp', f'Experiment-{exp_id}');
        print(f'\nsave to => {path_to_sum_wr}');
        print('*********\n');
    
    else:
        path_to_sum_wr = args.resume_path;

    return path_to_sum_wr;


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='SSLMRI Training', allow_abbrev=False);
    parser.add_argument('--batch-size', default=4, type=int);
    parser.add_argument('--WHU-path', default='Building change detection dataset', type=str);
    parser.add_argument('--tile-h', default=256, type=int, help='size to of each input patch');
    parser.add_argument('--tile-w', default=256, type=int, help='size to of each input patch');
    parser.add_argument('--learning-rate', default=1e-4, type=float);
    parser.add_argument('--deterministic', default=False, action='store_true', help='if we want to have same augmentation and same datae, for sanity check');
    parser.add_argument('--virtual-batch-size', default=1, type=int, help='use it if batch size does not fit GPU memory');
    parser.add_argument('--num-workers', default=0, type=int, help='num workers for data loader, should be equal to number of CPU cores');
    parser.add_argument('--bl-multiplier', default=10, type=int, help='boundary loss coefficient');
    parser.add_argument('--device', default='cuda', type=str, help='device to run models on');
    parser.add_argument('--debug-train-data', default=False, action='store_true', help='debug training data for debugging purposes');
    parser.add_argument('--pretrained', default=False, action='store_true', help='indicate wether to initalize with self-supervised pretrained  model or not');
    parser.add_argument('--pretraining', default=False, action='store_true', help='indicate if we are doing self-supervised pretraining (True) or training of segmenation model (False)');
    parser.add_argument('--fold', default=0, type=int, help='which fold to train and test model on');
    parser.add_argument('--network', default='VNET', type=str, help='which model to use, SWINUNETR is the other option');
    parser.add_argument('--pretrain-path', default='best_model-278.ckpt', type=str, help='path to self-supervised pretrain model');
    parser.add_argument('--resume', default=False, action='store_true',  help='inidcate wether we are training from scratch or resume training');
    parser.add_argument('--resume-path', default='exp/Experiment-6285152', type=str,  help='path to resume data');
    parser.add_argument('--cache-WHU-data', default=False, action='store_true',  help='if true, it caches train, valid and test dataset');
    parser.add_argument('--num-cache-data', default=200,  help='number of examples to cache for testing of self-supervised pretraining');
    parser.add_argument('--patch-size', default=32,  help='patch size for ViT');
    parser.add_argument('--diff-thresh', default=0.25, type=float, help='difference threshold');
    parser.add_argument('--mask-loss-multiplier', default=1, type=float, help='multiplier for the mask loss');
    parser.add_argument('--epoch', default=500, type=int);
    parser.add_argument('--dataset-size', default='one', help='if "all" use all the available samples in train and test set, if "one" only used one for each set, it is used for debugging purposes');
    parser.add_argument('--num-inpaint', default=20, type=int, help='number of inpainted region per cropped patch');
    parser.add_argument('--loss-type', default=0, type=int, help='type of loss function for consistency loss');
    parser.add_argument('--regularize-weight', default=1.0, type=float, help='weight of the regularizer or the consistency loss');
    parser.add_argument('--pretraining-scheme', default='MAE', help='chose between our novel inpainting task and MAE');
    parser.add_argument('--train-layers', default='all', help="what layers to train, for fine tuning only certrain layers we can use 'only-last' or 'only-decoder' or to train all we can use 'all' ");
    parser.add_argument('--sample-per-mri', default=8, help='how many crops from one MRI, only used during segmentation training');
    parser.add_argument('--ssl-head-type', default=2, type=int, help='type of ssl head for pretraining');
    parser.add_argument('--ssl-head-hidden-dim', default=256,type=int, help='type of ssl head for pretraining');
    parser.add_argument('--ssl-head-bottleneck-dim', default=128, type=int,help='type of ssl head for pretraining');
    parser.add_argument('--ssl-head-use-bn', default=False, action='store_true', help='type of ssl head for pretraining');
    parser.add_argument('--ssl-head-n-layers', default=3,type=int, help='type of ssl head for pretraining');
    
    args = parser.parse_args();

    if args.cache_WHU_data:
        cache_dataset_WHU(args);

    path_to_sum_wr = log_hyperparameters(args);
    if args.pretraining:
        if args.pretraining_scheme == 'inpainting':
            if args.network == 'VNET':
                model = VNet(args, model_type='pretraining', n_channels=3, n_classes=1, normalization='batchnorm', has_dropout=True, ssl_head_type=args.ssl_head_type).to(args.device)

        elif args.pretraining_scheme == 'MAE':
            model = MaskedAutoencoderViT()

    else:
        if args.network == 'VNET':
            model = VNet(args, model_type='segmentation', n_channels=6, n_classes=1, normalization='batchnorm', has_dropout=True).to(args.device)
        elif args.network == 'VNETCrossAttention':
            model = VNetCrossAttention(args, model_type='segmentation', n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=True).to(args.device)
        elif args.network == 'ViT':
            model = UNETR(in_channels=2, out_channels=1, img_size=96);
        else:
            model = SwinUNETR(img_size=(96,96,96), spatial_dims=2, in_channels=3, out_channels=1, feature_size=48).to(args.device)
            ckpt = torch.load('pretrained/swin/model_swinvit.pt');
            model.load_from(ckpt)

        if args.pretrained:
            ckpt = torch.load(args.pretrain_path);
            model.load_state_dict(ckpt['model'], strict=False); #strict = False because ssl_head is included in the checkpoint and we want to get rid of it

    
    if args.resume is True:
        ckpt = torch.load(os.path.join(path_to_sum_wr, 'resume.ckpt'));
        model.load_state_dict(ckpt['model']);

    model.to(args.device);
    scalar = torch.cuda.amp.grad_scaler.GradScaler();
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate);

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min= 1e-5);
    summary_writer = SummaryWriter(path_to_sum_wr);
    if args.pretraining:
        best_loss = 10;
    else:
        best_loss = 0;
    start_epoch = 0;

    if args.resume is True:
        optimizer.load_state_dict(ckpt['optimizer']);
        lr_scheduler.load_state_dict(ckpt['scheduler']);
        best_loss = ckpt['best_loss'];
        start_epoch = ckpt['epoch'];
        print(f'Resuming from epoch:{start_epoch}');

    if args.pretraining:
        train_loader, test_loader = get_loader_WHU(args);
    else:
        train_loader, test_ids = get_loader_WHU(args);
    
    sample_output_interval = 5;

    for epoch in range(start_epoch, args.epoch):
        model.train();
        if args.pretraining:
            train_loss = pretrain_step(args, model, train_loader, optimizer, scalar); 
        else:
            train_loss = train_step(args, model, train_loader, optimizer, scalar); 
       
        model.eval();
        if args.pretraining:
            valid_loss = pretrain_valid_step(args, model, test_loader);
        else:
            valid_loss = valid_step(args, model, test_ids);
        
        summary_writer.add_scalar('train/loss', train_loss, epoch);
        summary_writer.add_scalar('valid/loss', valid_loss, epoch);
        if epoch %sample_output_interval == 0 and args.pretraining:
            print('sampling outputs...');
            save_examples(model, test_loader,path_to_sum_wr);
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch+1
        }
        torch.save(ckpt,os.path.join(path_to_sum_wr, 'resume.ckpt'));
        #lr_scheduler.step();
        
        save_model = False;
        if args.pretraining:
            if best_loss > valid_loss:
                save_model = True
        else:
            if best_loss < valid_loss:
                save_model = True;
        
        if save_model:
            print(f'new best model found: {valid_loss}')
            best_loss = valid_loss;
            torch.save({'model': model.state_dict(), 
                        'best_loss': best_loss,
                        'log': path_to_sum_wr}, os.path.join(path_to_sum_wr, 'best_model.ckpt'));

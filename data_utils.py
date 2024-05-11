import os
from torch.utils.data import DataLoader, Dataset
import pickle
from glob import glob
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import nibabel as nib
from skimage.filters import sobel, threshold_otsu
from monai.transforms import Compose, Resize, Resize, RandGaussianSmooth, OneOf, RandGibbsNoise, RandGaussianNoise, GaussianSmooth, NormalizeIntensity, RandCropByPosNegLabeld, GibbsNoise, RandSpatialCropSamplesd
from scipy.ndimage import binary_opening
from tqdm import tqdm
import math
from patchify import patchify
from scipy.ndimage import distance_transform_edt, sobel, gaussian_filter
from utility import calculate_metric_percase
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def window_center_adjustment(img):
    """window center adjustment, similar to what ITKSnap does

    Parameters
    ----------
    img : np.ndarray
        input image

    """
    hist = np.histogram(img.ravel(), bins = int(np.max(img)))[0];
    hist = hist / (hist.sum()+1e-4);
    hist = np.cumsum(hist);

    hist_thresh = ((1-hist) < 5e-4);
    max_intensity = np.where(hist_thresh == True)[0][0];
    adjusted_img = img * (255/(max_intensity + 1e-4));
    adjusted_img = np.where(adjusted_img > 255, 255, adjusted_img).astype("uint8");

    return adjusted_img;

def tile_image(img, tile_w, tile_h, overlap = None):
    if img.ndim == 3:
        h,w,_ = img.shape;
    else:
        h,w = img.shape;

    #padd if not divisable by tile_h and tile_w
    pad_w =  (tile_w - (w%tile_w)) + w if w%tile_w !=0 else w;
    pad_h =  (tile_h - (h%tile_h)) + h if h%tile_h !=0 else h;
    
    if img.ndim == 3:
        img_padded = np.zeros((pad_h, pad_w, 3), dtype=img.dtype);
        img_padded[:h, :w, :] = img;
    else:
        img_padded = np.zeros((pad_h, pad_w), dtype=img.dtype);
        img_padded[:h, :w] = img;
    tiles = [];
    for x in range(0, pad_w-(tile_w if overlap == None else overlap), tile_w if overlap == None else overlap):
        for y in range(0, pad_h-(tile_h if overlap == None else overlap), tile_h if overlap == None else overlap):
            if img.ndim == 3:
                tiles.append(img_padded[y:y+tile_h, x:x+tile_w,:])
            else:
                tiles.append(img_padded[y:y+tile_h, x:x+tile_w])    
    return tiles;

def cache_dataset_WHU(args):
    """cache testing set for self-supervised pretraining model for WHU dataset

    Parameters
    ----------
    args : dict
        arguments

    """
    map_before = cv2.imread(os.path.join(args.WHU_path, '1. The two-period image data', 'before', 'before.tif'), cv2.IMREAD_COLOR);
    map_after = cv2.imread(os.path.join(args.WHU_path, '1. The two-period image data', 'after', 'after.tif'), cv2.IMREAD_COLOR);
    label = cv2.imread(os.path.join(args.WHU_path, '1. The two-period image data', 'change label', 'change_label.tif'), cv2.IMREAD_GRAYSCALE);
    h,w,_ = map_before.shape;

    train_w = int(w*.6);
    valid_w = train_w + int(w*.1);

    train_crop_after = map_after[:,:train_w,:];
    train_crop_before = map_before[:,:train_w,:];

    valid_crop_after = map_after[:,train_w:valid_w,:];
    valid_crop_before = map_before[:,train_w:valid_w,:];

    test_crop_after = map_after[:,valid_w:,:];
    test_crop_before = map_before[:,valid_w:,:];

    train_label = label[:,:train_w];
    valid_label = label[:,train_w:valid_w];
    test_label = label[:,valid_w:w];


    tiles_train_before = tile_image(train_crop_before, tile_w=args.tile_w, tile_h=args.tile_h, overlap=128);
    tiles_train_after = tile_image(train_crop_after, tile_w=args.tile_w, tile_h=args.tile_h, overlap=128);
    tiles_train_label = tile_image(train_label, tile_w=args.tile_w, tile_h=args.tile_h, overlap=128);

    tiles_valid_before = tile_image(valid_crop_before, tile_w=args.tile_w, tile_h=args.tile_h);
    tiles_valid_after = tile_image(valid_crop_after, tile_w=args.tile_w, tile_h=args.tile_h);
    tiles_valid_label = tile_image(valid_label, tile_w=args.tile_w, tile_h=args.tile_h);

    tiles_test_before = tile_image(test_crop_before, tile_w=args.tile_w, tile_h=args.tile_h);
    tiles_test_after = tile_image(test_crop_after, tile_w=args.tile_w, tile_h=args.tile_h);
    tiles_test_label = tile_image(test_label, tile_w=args.tile_w, tile_h=args.tile_h);

    if os.path.exists('cache-WHU') is False:
        os.mkdir('cache-WHU');
    
    pickle.dump(tiles_train_before, open(os.path.join('cache-WHU','tiles_train_before.dmp'), 'wb'));
    pickle.dump(tiles_train_after, open(os.path.join('cache-WHU', 'tiles_train_after.dmp'), 'wb'));
    pickle.dump(tiles_train_label, open(os.path.join('cache-WHU', 'tiles_train_label.dmp'), 'wb'));

    pickle.dump(tiles_valid_before, open(os.path.join('cache-WHU','tiles_valid_before.dmp'), 'wb'));
    pickle.dump(tiles_valid_after, open(os.path.join('cache-WHU', 'tiles_valid_after.dmp'), 'wb'));
    pickle.dump(tiles_valid_label, open(os.path.join('cache-WHU', 'tiles_valid_label.dmp'), 'wb'));

    pickle.dump(tiles_test_before, open(os.path.join('cache-WHU','tiles_test_before.dmp'), 'wb'));
    pickle.dump(tiles_test_after, open(os.path.join('cache-WHU', 'tiles_test_after.dmp'), 'wb'));
    pickle.dump(tiles_test_label, open(os.path.join('cache-WHU', 'tiles_test_label.dmp'), 'wb'));

    

def cropper(mri1, 
            mri2, 
            gt, 
            gr, 
            roi_size,
            num_samples):
    """crop two time-points MRI scans at the same time for new lesion segmentation model

    Parameters
    ----------
    mri1 : np.ndarray
        first MRI scan

    mri2 : np.ndarray
        second MRI scan

    gt : np.ndarray
        ground truth segmentations

    gr : np.ndarray
        gradients of MRI scan

    rot_size : list
        size to crop for three axis

    num_samples : int
        number of samples to crop

    """
    ret = [];
    for i in range(num_samples):
        pos_cords = np.where(gt > 0);
        r = np.random.randint(0,len(pos_cords[0]));
        center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
        d_x_l = min(roi_size[0]//2,center[0]);
        d_x_r = min(roi_size[0]//2 ,mri1.shape[1]-center[0]);
        if d_x_l != roi_size[0]//2:
            diff = abs(roi_size[0]//2 - center[0]);
            d_x_r += diff;
        if d_x_r != roi_size[0]//2 and d_x_l == roi_size[0]//2:
            diff = abs(roi_size[0]//2 - (mri1.shape[1]-center[0]));
            d_x_l += diff;
        
        d_y_l = min(roi_size[1]//2,center[1]);
        d_y_r = min(roi_size[1]//2 ,mri1.shape[2]-center[1]);
        if d_y_l != roi_size[1]//2:
            diff = abs(roi_size[1]//2 - center[1]);
            d_y_r += diff;
        if d_y_r != roi_size[1]//2 and d_y_l == roi_size[1]//2:
            diff = abs(roi_size[1]//2 - mri1.shape[2]-center[1]);
            d_y_l += diff;
        
        d_z_l = min(roi_size[2]//2,center[2]);
        d_z_r = min(roi_size[2]//2 ,mri1.shape[3]-center[2]);
        if d_z_l != roi_size[2]//2:
            diff = abs(roi_size[2]//2 - center[2]);
            d_z_r += diff;
        if d_z_r != roi_size[2]//2 and d_z_l == roi_size[2]//2:
            diff = abs(roi_size[2]//2 - mri1.shape[3]-center[2]);
            d_z_l += diff;

        sign_x = np.random.randint(1,3);
        if sign_x%2!=0:
            offset_x = np.random.randint(0, max(min(abs(center[0]-int(d_x_l)), int(d_x_l//2)),1))*-1;
        else:
            offset_x = np.random.randint(0, max(min(abs(center[0]+int(d_x_r)-mri1.shape[1]), int(d_x_r//2)), 1));
        start_x = center[0]-int(d_x_l)+offset_x;
        end_x = center[0]+int(d_x_r)+offset_x;

        sign_y = np.random.randint(1,3);
        if sign_y%2!=0:
            offset_y = np.random.randint(0, max(min(abs(center[1]-int(d_y_l)), int(d_y_l//2)),1))*-1;
        else:
            offset_y = np.random.randint(0, max(min(abs(center[1]+int(d_y_r)-mri1.shape[2]), int(d_y_r//2)), 1));
        start_y = center[1]-int(d_y_l) + offset_y;
        end_y = center[1]+int(d_y_r) + offset_y;

        sign_z = np.random.randint(1,3);
        if sign_z%2!=0:
            offset_z = np.random.randint(0, max(min(abs(center[2]-int(d_z_l)), int(d_z_l)),1))*-1;
        else:
            offset_z = np.random.randint(0, max(min(abs(center[2]+int(d_z_r)-mri1.shape[3]), int(d_z_r//2)), 1));
        
        start_z = center[2]-int(d_z_l)+offset_z;
        end_z = center[2]+int(d_z_r)+offset_z;

        d = dict();
        d['image1'] = torch.from_numpy(mri1[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['image2'] = torch.from_numpy(mri2[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['mask'] = torch.from_numpy(gt[:, start_x:end_x, start_y:end_y, start_z:end_z]);
        d['gradient'] = torch.from_numpy(gr[:,start_x:end_x, start_y:end_y, start_z:end_z]);

        ret.append(d);

    return ret;

class MICCAI_PRETRAIN_Dataset(Dataset):
    """Dataset for self-supervised pretraining

    it returns examples for training which includes to MRI patch and one ground truth labels

    Parameters
    ----------
    args : dict
        arguments

    mr_images : list
        list of mri images, should a string list

    train : bool
        indicate if we are in training or testing mode

    cache: bool
        inidicate if we are only caching dataset or we are using it for training

    Attributes
    ----------
    mr_imges : list
        list of loaded mri scans

    """
    def __init__(self, 
                 args, 
                 mr_images, 
                 train = True,
                 cache = False,
                 cache_sample_per_mri = 8) -> None:
        super().__init__();

        self.args = args;
        m1 = 0.7;
        m2 = 0.8;
        self.augment_noisy_image = OneOf([
            RandGaussianSmooth(prob=1.0, sigma_x=(m1,m2), sigma_y=(m1,m2), sigma_z=(m1,m2)),
            RandGaussianNoise(prob=1.0,std=0.05),
            RandGibbsNoise(prob=1.0, alpha=(0.65,0.75))
        ], weights=[1,1,1])


        self.transforms = NormalizeIntensity(subtrahend=0.5, divisor=0.5);
        self.crop = RandCropByPosNegLabeld(
            keys=['image', 'mask'], 
            label_key='mask', 
            spatial_size= (args.crop_size_w, args.crop_size_h, args.crop_size_d),
            pos=1, 
            neg=0,
            num_samples=1);
        
        self.resize = Resize(spatial_size=[args.crop_size_w, args.crop_size_h, args.crop_size_d]);

        self.train = train;
        self.cache = cache;
        self.cache_sample_per_mri = cache_sample_per_mri;

        self.mr_images = [];

        if train or cache:
            for patient_path in mr_images:
                base_path = os.path.dirname(patient_path);
                base_path = base_path[:base_path.rfind('/')];

                mri = nib.load(patient_path);
                mri = mri.get_fdata();
                mri = window_center_adjustment(mri);

                brain_mask = nib.load(os.path.join(base_path, 'Masks', 'Brain_Mask.nii.gz'));
                brain_mask = brain_mask.get_fdata();

                self.mr_images.append([mri, brain_mask[None,:,:,:]])

        else:
            for mr in mr_images:
                mrimage = pickle.load(open(mr, 'rb'))
                self.mr_images.append(mrimage);
    
    def __len__(self):
        return len(self.mr_images);

    def __get_train(self, index):
        mrimage, mask = self.mr_images[index];
        mrimage = np.expand_dims(mrimage, axis=0);
        mrimage = mrimage / (np.max(mrimage)+1e-4);

        if self.args.deterministic is False:
            ret_transforms = self.crop({'image': mrimage, 'mask': mask});
        
        if self.args.deterministic is False:
            mrimage_c = ret_transforms[0]['image'];
            mask_c = ret_transforms[0]['mask'];
            mrimage_noisy = copy(mrimage_c);
        else:
            mrimage_c = mrimage[:, int(mrimage.shape[1]/2-48):int(mrimage.shape[1]/2+48), int(mrimage.shape[2]/2-48):int(mrimage.shape[2]/2+48), int(mrimage.shape[3]/2-48):int(mrimage.shape[3]/2+48)];
            mask_c = mask[:, int(mrimage.shape[1]/2-48):int(mrimage.shape[1]/2+48), int(mrimage.shape[2]/2-48):int(mrimage.shape[2]/2+48), int(mrimage.shape[3]/2-48):int(mrimage.shape[3]/2+48)];
            mrimage_noisy = copy(mrimage_c);
            mrimage_c = torch.from_numpy(mrimage_c);
            mrimage_noisy = torch.from_numpy(mrimage_noisy);
            mask_c = torch.from_numpy(mask_c);
        
        num_corrupted_patches = self.args.num_inpaint if self.args.deterministic is False else 20;

        mrimage_noisy, heatmap, noise, center = inpaint_3d(mrimage_noisy, mask_c, num_corrupted_patches, self.args.deterministic)

        
        #total_heatmap = total_heatmap * mask_c;
        mrimage_noisy = mrimage_noisy * mask_c;
        #mrimage_c = mrimage_c * mask_c;

        total_heatmap_thresh = torch.where(heatmap > 0, 1.0, 0.0);
        part_first = mrimage_c * total_heatmap_thresh;
        part_second = mrimage_noisy * total_heatmap_thresh;

        if self.args.deterministic is True:
            mrimage_noisy = GibbsNoise(alpha = 0.65)(mrimage_noisy);
        else:
            mrimage_noisy = self.augment_noisy_image(mrimage_noisy);

        diff = torch.abs(part_first - part_second) > (self.args.diff_thresh);

        total_heatmap_thresh = torch.where(diff > 0, 0, 1);
        # c = torch.nn.Conv3d(1, 1, self.args.patch_size, self.args.patch_size, bias = False);
        # c.requires_grad_ = False;
        # c.weight.data = torch.ones_like(c.weight.data);
        # patched = c((1-total_heatmap_thresh.unsqueeze(dim=0)).float());
        # patched = torch.where(patched > (self.args.patch_size**3) / 4, 1, 0);
        # patched = torch.nn.functional.upsample(patched.float(), (96,96,96));
        
        if self.args.debug_train_data:
            visualize_2d([mrimage_c, mrimage_noisy, 1-total_heatmap_thresh, mask_c, noise], center);
        
        mrimage_c = self.transforms(mrimage_c)[0];
        mrimage_noisy = self.transforms(mrimage_noisy)[0];

        # if ret_mrimage is None:
        #     ret_mrimage = mrimage_c.unsqueeze(dim=0);
        #     ret_mrimage_noisy = mrimage_noisy.unsqueeze(dim=0);

        #     ret_total_heatmap = total_heatmap_thresh.unsqueeze(dim=0);
        # else:
        #     ret_mrimage = torch.concat([ret_mrimage, mrimage_c.unsqueeze(dim=0)], dim=0);
        #     ret_mrimage_noisy = torch.concat([ret_mrimage_noisy, mrimage_noisy.unsqueeze(dim=0)], dim=0);

        #     ret_total_heatmap = torch.concat([ret_total_heatmap, total_heatmap_thresh.unsqueeze(dim=0)], dim=0);
    
    # with torch.no_grad():
    #     c = torch.nn.Conv3d(1, 1, self.args.patch_size, self.args.patch_size, bias = False);
    #     c.requires_grad_ = False;
    #     c.weight.data = torch.ones_like(c.weight.data);
    #     patched = c((1-ret_total_heatmap).float());
    #     patched = torch.where(patched >  (self.args.patch_size**3) / 4, 1, 0);


            
        return mrimage_c.unsqueeze(0), mrimage_noisy.unsqueeze(0), total_heatmap_thresh;

    def __getitem__(self, index):
        if self.train or self.cache:
            return self.__get_train(index);
        else:
            ret = self.mr_images[index];
            return ret;

class WHU_Dataset(Dataset):
    """Dataset for WHU dataset

    it returns examples for training, testing and validation

    Parameters
    ----------
    args : dict
        arguments

    after : list
        after tiles

    before : bool
        befor tiles
    
    label : bool
        label for changes happened from before to after
    
    train: bool
        indicate whether we are loading data for training or not

    """
    def __init__(self, 
                args,
                before,
                after,
                label,
                train = False) -> None:
        super().__init__();

        self.args = args;
        self.after = after;
        self.before = before;
        self.label = label;
        self.train = train;
    
        self.train_aug = A.Compose([
            A.Equalize(),
            A.Flip(),
            A.Normalize(),
            ToTensorV2()
        ],
            additional_targets={
                'image0' : 'image',
                'mask0' : 'mask'
            }
        )

        self.valid_aug = A.Compose([
            A.Equalize(),
            A.Normalize(),
            ToTensorV2()
        ],
            additional_targets={
                'image0' : 'image',
                'mask0' : 'mask'
            }
        )

    def __len__(self):
        return len(self.before);

    def __getitem__(self, index):
        if self.train:
            before, after, label = self.before[index], self.after[index], self.label[index];
            label = np.where(label == 255, 1, 0);
            count_nonzero = np.count_nonzero(label);
            aug_out = self.train_aug(image = before, image0 = after, mask0 = label);

            before = aug_out['image'];
            after = aug_out['image0'];
            label = aug_out['mask0'];

            pos_dt = distance_transform_edt(np.where(label.squeeze()==1, 0, 1));
            pos_dt = pos_dt/(np.max(pos_dt)+1e-4);

            neg_dt = distance_transform_edt(label.squeeze()==1);
            neg_dt = neg_dt/(np.max(neg_dt)+1e-4);

            dt = pos_dt - neg_dt;
            dt = torch.from_numpy(np.expand_dims(dt, axis = 0));

            #if we have a  zero label, make a constant one distance transform
            if count_nonzero == 0:
                dt = torch.ones_like(dt);

            if self.args.debug_train_data:
                fig, ax = plt.subplots(1, 4);
                ax[0].imshow(before.permute(1,2,0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]);
                ax[1].imshow(after.permute(1,2,0).numpy()  * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]);
                ax[2].imshow(label);
                ax[3].imshow(dt.squeeze().numpy());
                plt.show();
            
            return before, after, label.unsqueeze(0), dt;
        else:
            before, after, label = self.before[index], self.after[index], self.label[index];
            aug_out = self.valid_aug(image = before, image0 = after, mask0 = label);

            before = aug_out['image'];
            after = aug_out['image0'];
            label = aug_out['mask0'];

            return before, after, label.unsqueeze(0);

def get_loader_WHU(args):
    """prepare train and test loader for training with WHU dataset

        Parameters
        ----------
        args : dict
            arguments.
    """
    train_before = pickle.load(open(os.path.join('cache-WHU', f'tiles_train_before.dmp'), 'rb'));
    train_after = pickle.load(open(os.path.join('cache-WHU', f'tiles_train_after.dmp'), 'rb'));
    train_label = pickle.load(open(os.path.join('cache-WHU', f'tiles_train_label.dmp'), 'rb'));

    valid_before = pickle.load(open(os.path.join('cache-WHU',f'tiles_valid_before.dmp'), 'rb'));
    valid_after = pickle.load(open(os.path.join('cache-WHU', f'tiles_valid_after.dmp'), 'rb'));
    valid_label = pickle.load(open(os.path.join('cache-WHU', f'tiles_valid_label.dmp'), 'rb'));
    
    test_before = pickle.load(open(os.path.join('cache-WHU',f'tiles_test_before.dmp'), 'rb'));
    test_after = pickle.load(open(os.path.join('cache-WHU', f'tiles_test_after.dmp'), 'rb'));
    test_label = pickle.load(open(os.path.join('cache-WHU', f'tiles_test_label.dmp'), 'rb'));


    train_dataset = WHU_Dataset(args, 
                                    train_before if args.dataset_size == 'all' else train_before[:1],
                                    train_after if args.dataset_size == 'all' else train_after[:1],
                                    train_label if args.dataset_size == 'all' else train_label[:1],
                                    train = True);
    
    valid_dataset = WHU_Dataset(args, 
                                    valid_before if args.dataset_size == 'all' else valid_before[:1],
                                    valid_after if args.dataset_size == 'all' else valid_after[:1],
                                    valid_label if args.dataset_size == 'all' else valid_label[:1]);
    
    
    train_loader = DataLoader(train_dataset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True);
    valid_loader = DataLoader(valid_dataset, args.batch_size, False, num_workers=args.num_workers, pin_memory=True);

    return train_loader, valid_loader; 

def get_loader_miccai(args, fold):
    """prepare train and test loader for new lesion segmentation model

        Parameters
        ----------
        args : dict
            arguments.
    """
    with open(os.path.join('cache_miccai', f'fold{fold}.txt'), 'r') as f:
        train_ids = f.readline().rstrip();
        train_ids = train_ids.split(',');
        test_ids = f.readline().rstrip();
        test_ids = test_ids.split(',');
    train_ids =  [os.path.join('miccai-processed', t) for t in train_ids];
    test_ids = [os.path.join('miccai-processed', t) for t in test_ids];


    mri_dataset_train = MICCAI_Dataset(args, train_ids if args.dataset_size == 'all' else train_ids[:1], train=True);
    train_loader = DataLoader(mri_dataset_train, 1, True, num_workers=args.num_workers, pin_memory=True);
    mri_dataset_test = MICCAI_Dataset(args, test_ids if args.dataset_size == 'all' else test_ids[:1], train=False);
    test_loader = DataLoader(mri_dataset_test, 1, False, num_workers=args.num_workers, pin_memory=True);

    return train_loader, test_loader, mri_dataset_test; 

def visualize_2d(images, slice,):
    """display one slice of an MRI scan, for debugging purposes

        Parameters
        ----------
        args : dict
            arguments.
    """
    fig, ax = plt.subplots(len(images),3);
    for i,img in enumerate(images):
        img = img.squeeze();
        ax[i][0].imshow(img[slice[0], :,:], cmap='gray');
        ax[i][1].imshow(img[:,slice[1],:], cmap='gray');
        ax[i][2].imshow(img[:,:,slice[2]], cmap='gray');
    plt.show()

def inpaint_3d(img, 
               mask_g, 
               num_corrupted_patches, 
               deterministic = False):
    """remove part of an MRI scan for self-superivsed pretraining model

        Parameters
        ----------
        img : np.ndarray
            MRI scan patch

        mask_g : np.ndarray
            mask to take inpainting centers from

        num_corrupted_patches : int
            number of patches to curropt

        deterministic : bool
            if true, every call to this function yields the same results.
    """
    mri = img;

    _,h,w,d = mri.shape;

    cube = np.zeros((1,h,w,d), dtype=np.uint8);
    mask_cpy = deepcopy(mask_g);
    for n in range(num_corrupted_patches):
        size_x = np.random.randint(5,15,) if deterministic is False else 15;
        size_y = np.random.randint(5,20) if deterministic is False else 15;
        size_z = np.random.randint(5,20) if deterministic is False else 15;

        mask_g[:,:,:,d-size_z:] = 0;
        mask_g[:,:,:,:size_z+1] = 0;
        mask_g[:,:,w-size_y:,:] = 0;
        mask_g[:,:,:size_y+1,:] = 0;
        mask_g[:,h-size_x:,:,:] = 0;
        mask_g[:,:size_x+1,:,:] = 0;
        pos_cords = np.where(mask_g==1);

        if deterministic is False:
            if len(pos_cords[0]) != 0:
                r = np.random.randint(0,len(pos_cords[0]));
                center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
            else:
                center = [img.shape[0]//2, img.shape[1]//2, img.shape[2]//2]
        else:
            center = [pos_cords[1][50*(n+1)], pos_cords[2][50*(n+1)],pos_cords[3][50*(n+1)]]
        cube[:,max(center[0]-size_x,0):min(center[0]+size_x, h), \
             max(center[1]-size_y,0):min(center[1]+size_y,w), \
             max(center[2]-size_z,0):min(center[2]+size_z,d)] = 1;
        mask_g[:,:,:,d-size_z:] = mask_cpy[:,:,:,d-size_z:];
        mask_g[:,:,:,:size_z+1] = mask_cpy[:,:,:,:size_z+1];
        mask_g[:,:,w-size_y:,:] = mask_cpy[:,:,w-size_y:,:];
        mask_g[:,:,:size_y+1,:] = mask_cpy[:,:,:size_y+1,:];
        mask_g[:,h-size_x:,:,:] = mask_cpy[:,h-size_x:,:,:];
        mask_g[:,:size_x+1,:,:] = mask_cpy[:,:size_x+1,:,:];
    
    #shape
    # for c in cubes:
    #     cube[:,max(c[0][0]-c[1],0):min(c[0][0]+c[1], h), \
    #          max(c[0][1]-c[2],0):min(c[0][1]+c[2],w), \
    #          max(c[0][2]-c[3],0):min(c[0][2]+c[3],d)] = 1;

    #cube = transform(cube);
    cube_thresh = (cube>0)

    smoothness = np.random.randint(3,7)
    cube_thresh = GaussianSmooth(smoothness, approx='erf')(cube_thresh);
    cube_thresh = cube_thresh / (torch.max(cube_thresh) + 1e-4);
    #================

    smoothness = np.random.randint(15,20)
    noise = GaussianSmooth(smoothness, )(mri);
    mri_after = (1-cube_thresh)*mri + (cube_thresh*noise);
    #noise = GaussianSmooth(7)(mask_g.float());
    
    #mri_after = torch.clip(mri_after, 0, 1);
    #mri_after = (mri_after*255).astype("uint8")
    #visualize_2d(mri_after, cube_thresh, slice=center[0:]);
    return mri_after, cube_thresh, noise, center;

def add_synthetic_lesion_wm(img, 
                            mask_g, 
                            deterministic):
    """adds synthetic lesions to the MRI scan

        Parameters
        ----------
        img : np.ndarray
            MRI scan patch

        mask_g : np.ndarray
            mask to take inpainting centers from

        deterministic : bool
            if true, every call to this function yields the same results.
    """
    mri = img;

    _,h,w,d = mri.shape;

    mask_cpy = deepcopy(mask_g);
    size_x = np.random.randint(2,6) if deterministic is False else 3;
    size_y = size_x - np.random.randint(0,size_x-1) if deterministic is False else 3;
    size_z = size_x - np.random.randint(0,size_x-1) if deterministic is False else 3;
    mask_cpy[:,:,:,d-size_z:] = 0;
    mask_cpy[:,:,:,:size_z+1] = 0;
    mask_cpy[:,:,w-size_y:,:] = 0;
    mask_cpy[:,:,:size_y+1,:] = 0;
    mask_cpy[:,h-size_x:,:,:] = 0;
    mask_cpy[:,:size_x+1,:,:] = 0;
    pos_cords = np.where(mask_cpy==1);

    if deterministic is False:
        if len(pos_cords[0]) != 0:
            r = np.random.randint(0,len(pos_cords[0]));
            center = [pos_cords[1][r], pos_cords[2][r],pos_cords[3][r]]
        else:
            center = [img.shape[1]//2, img.shape[2]//2, img.shape[3]//2]
    else:
        if len(pos_cords[0]) != 0:
            center = [pos_cords[1][int(len(pos_cords[0])//2)], pos_cords[2][int(len(pos_cords[0])//2)],pos_cords[3][int(len(pos_cords[0])//2)]]
        else:
            center = [img.shape[1]//2, img.shape[2]//2, img.shape[3]//2]
    
 
    #shape
    cube = torch.zeros((1,h,w,d), dtype=torch.uint8);
    cube[:,max(center[0]-size_x,0):min(center[0]+size_x, h), max(center[1]-size_y,0):min(center[1]+size_y,w), max(center[2]-size_z,0):min(center[2]+size_z,d)] = 1;
    cube = cube * mask_g;

    cube = GaussianSmooth(1.2, approx='erf')(cube);
    cube = cube / (torch.max(cube) + 1e-4);
    #================

    noise = (torch.ones((1,h,w,d), dtype=torch.uint8));
    final = (cube)*(noise);
    mri_after = (1-cube)*mri + final;
    
    return mri_after, cube;
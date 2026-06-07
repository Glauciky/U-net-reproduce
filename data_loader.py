import os,glob
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
class PolypDataset(Dataset):
    def __init__(self,img_dir,mask_dir,transform=None,img_size=572,roi_size=388):
        self.img_paths=sorted(glob.glob(os.path.join(img_dir,'*')))
        self.mask_paths=[os.path.join(mask_dir,os.path.basename(p)) for p in self.img_paths]
        self.transform=transform
        self.img_size=img_size
        self.roi_size=roi_size
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,idx):
        img=cv2.imread(self.img_paths[idx])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(self.mask_paths[idx],cv2.IMREAD_GRAYSCALE)
        mask=(mask>127).astype(np.uint8)*255
        h,w=img.shape[:2]
        scale=self.roi_size/max(h,w)
        new_h=int(round(h*scale))
        new_w=int(round(w*scale))
        img=cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize(mask,(new_w,new_h),interpolation=cv2.INTER_NEAREST)
        canvas_img=cv2.copyMakeBorder(
            img,
            top=(self.img_size-new_h)//2,
            bottom=self.img_size-new_h-(self.img_size-new_h)//2,
            left=(self.img_size-new_w)//2,
            right=self.img_size-new_w-(self.img_size-new_w)//2,
            borderType=cv2.BORDER_REFLECT_101
        )
        canvas_mask=cv2.copyMakeBorder(
            mask,
            top=(self.img_size-new_h)//2,
            bottom=self.img_size-new_h-(self.img_size-new_h)//2,
            left=(self.img_size-new_w)//2,
            right=self.img_size-new_w-(self.img_size-new_w)//2,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )
        crop_start=(self.img_size-self.roi_size)//2
        if self.transform:
            augmented=self.transform(image=canvas_img,mask=canvas_mask)
            canvas_img=augmented['image']
            canvas_mask=augmented['mask']
            mask_target=canvas_mask[crop_start:crop_start+self.roi_size,crop_start:crop_start+self.roi_size]
            mask_target=(mask_target.float()/255.0>0.5).float() #【修改】统一除255后再二值化，与else分支对齐
            mask_target=mask_target.unsqueeze(0)
        else:
            canvas_img=torch.from_numpy(canvas_img.transpose(2,0,1)).float()/255.0
            mask_target=canvas_mask[crop_start:crop_start+self.roi_size,crop_start:crop_start+self.roi_size]
            mask_target=torch.from_numpy(mask_target).float()/255.0
            mask_target=(mask_target>0.5).float() #【修改】加二值化，与transform分支统一输出干净的0/1
            mask_target=mask_target.unsqueeze(0)
        return canvas_img,mask_target
#数据增强
def get_dataloaders(batch_size=4,data_root="dataset"): 
    train_transform=A.Compose([
        A.HorizontalFlip(p=0.5),       #水平翻转
        A.VerticalFlip(p=0.5),     #垂直翻转
        A.RandomRotate90(p=0.5),#90的倍数随机翻转
        A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5), #亮度对比度
        A.GaussianBlur(blur_limit=(3,5),p=0.3),           #模拟模糊
        A.GaussNoise(
        std_range=(0.01, 0.03),
        mean_range=(0.0, 0.0),
        p=0.2),                     #噪声
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]), #简便化处理
        ToTensorV2()
    ])
    val_transform=A.Compose([
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    #设置数据集
    train_set=PolypDataset(
        os.path.join(data_root,"train/images"),
        os.path.join(data_root,"train/masks"),
        transform=train_transform
    )
    val_set=PolypDataset(
        os.path.join(data_root,"val/images"),
        os.path.join(data_root,"val/masks"),
        transform=val_transform
    )
    test_set=PolypDataset(
        os.path.join(data_root,"test/images"),
        os.path.join(data_root,"test/masks"),
        transform=val_transform
    )
    train_loader=DataLoader(train_set,batch_size=batch_size,
                            shuffle=True,pin_memory=True,num_workers=2)
    val_loader=DataLoader(val_set,batch_size=batch_size,
                          shuffle=False,pin_memory=True,num_workers=2)
    test_loader=DataLoader(test_set,batch_size=batch_size,
                           shuffle=False,pin_memory=True,num_workers=2)
    return train_loader,val_loader,test_loader
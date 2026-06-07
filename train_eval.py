import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np 
class DiceLoss(nn.Module):
    def __init__(self,smooth=1.0): 
        super().__init__()
        self.smooth=smooth
    def forward(self,inputs,targets):
        inputs=torch.sigmoid(inputs)
        inter=(inputs*targets).sum(dim=(2,3))
        union=inputs.sum(dim=(2,3))+targets.sum(dim=(2,3))
        dice=(2*inter+self.smooth)/(union+self.smooth)
        return 1-dice.mean()
#修改加入混合loss，BCE和Dice组合
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.8,bce_weight=0.2):
        super().__init__()
        self.dice=DiceLoss()
        self.bce=nn.BCEWithLogitsLoss()
        self.dice_w=dice_weight
        self.bce_w=bce_weight
    def forward(self, pred, target):
        dice_loss=self.dice(pred, target)
        bce_loss=self.bce(pred, target)
        combine_loss=(self.dice_w*dice_loss+self.bce_w*bce_loss)
        return combine_loss
def train_epoch(model,train_loader,optimizer,criterion,device):
    model.train()
    total_loss=0
    #增加进度条显示
    loader = tqdm(train_loader, desc="Training", leave=False)
    for imgs,masks in loader:
        imgs,masks=imgs.to(device),masks.to(device)
        optimizer.zero_grad() 
        preds=model(imgs)
        loss=criterion(preds,masks)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        loader.set_postfix(loss=loss.item())
    return total_loss/len(loader)
#保存对比图
def save_prediction_visualization(images, masks, preds, epoch,save_dir="vis",
                                  mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)):
    os.makedirs(save_dir, exist_ok=True)
    #可以修改查看的图片
    index=0
    image=images[index].cpu().permute(1, 2, 0).numpy()
    crop_start=(572-388)//2
    image=image[
        crop_start:crop_start+388,
        crop_start:crop_start+388]
    mean_arr=np.array(mean)
    std_arr=np.array(std)
    image=image*std_arr+mean_arr
    image=np.clip(image,0,1)
    gt=masks[index].cpu().squeeze().numpy()
    pred=preds[index].cpu().squeeze().numpy()
    _,axes = plt.subplots(1, 4, figsize=(16, 4))
#原图
    axes[0].imshow(image)
    axes[0].set_title("Image")
#MASK
    axes[1].imshow(gt,cmap='gray')
    axes[1].set_title("Ground Truth")
#预测图
    axes[2].imshow(pred,cmap='gray',vmin=0,vmax=1)
    axes[2].set_title("Prediction")
    axes[3].imshow(image)
    axes[3].imshow(pred>0.35,cmap='jet',alpha=0.5)
    axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.close()
#Val测试
def evaluate(model, loader, device, epoch=0):
    model.eval()
    dices=[]
    smooth=1e-6

    with torch.no_grad():
        for batch_idx,(imgs,masks) in enumerate(loader):
            imgs, masks=imgs.to(device),masks.to(device)

            preds=model(imgs)
            probs=torch.sigmoid(preds)
            preds_binary=(probs > 0.5).float()

            if batch_idx==0:
                save_prediction_visualization(
                    imgs,
                    masks,
                    probs,
                    epoch)
            inter=(preds_binary*masks).sum(dim=(1,2,3))
            union=preds_binary.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
            dice=(2*inter+smooth)/(union+smooth)
            dices.extend(dice.cpu().numpy())
    return np.mean(dices)
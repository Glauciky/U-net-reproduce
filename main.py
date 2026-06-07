import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os,copy
from unet_model import UNet
from data_loader import get_dataloaders
from train_eval import CombinedLoss,train_epoch,evaluate
def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    model=UNet(in_channels=3,out_channels=1).to(device)
    start_epoch=0
    best_dice=0.0
    #保存路径
    checkpoint_path='best_model.pth'
    #加载旧的模型数据
    if os.path.exists(checkpoint_path):
        ckpt=torch.load(checkpoint_path,map_location=device)
        model.load_state_dict(ckpt['model'])
        best_dice=ckpt.get('best_dice',0.0)
        print(f"加载预训练模型,best_dice={best_dice:.4f},从 epoch {start_epoch} 开始")
    train_loader,val_loader,test_loader=get_dataloaders(batch_size=4,data_root="dataset")
    criterion=CombinedLoss(dice_weight=0.7,bce_weight=0.3).to(device)
    #AdamW优化权重
    optimizer=optim.AdamW(model.parameters(),lr=1.5e-4,weight_decay=1e-4)
    scheduler=CosineAnnealingWarmRestarts(optimizer,T_0=20,T_mult=1,eta_min=1e-6)
    epochs=150
    #停止阈值
    patience=50
    no_improve=0
    for epoch in range(epochs):
        train_loss=train_epoch(model,train_loader,optimizer,criterion,device)
        val_dice=evaluate(model,val_loader,device,epoch)
        scheduler.step() 
        lr=optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {lr:.6f}")
        #保存Dice最高的模型
        if val_dice>best_dice:
            best_dice=val_dice
            best_wts=copy.deepcopy(model.state_dict())
            torch.save({
                'epoch':epoch,
                'model':best_wts,
                'best_dice':best_dice,
                'optimizer':optimizer.state_dict(),
            },checkpoint_path)
            no_improve=0
            print(f"保存最佳模型 (Dice: {val_dice:.4f})")
        else:
            #patience轮没有更新则早点结束
            no_improve+=1
            if no_improve>=patience:
                print(f"早停于 epoch {epoch+1}")
                break
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    #test数据集与train，val不是同一批仅作为参考
    test_dice=evaluate(model,test_loader,device)
    print(f"测试集 Dice: {test_dice:.4f}")
if __name__=="__main__":
    main()
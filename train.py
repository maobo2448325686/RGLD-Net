from utils.dataset import ISBI_Loader
from torch import optim, nn
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
import torch
import glob
import os
from model.BENet import BENet
from utils.evaluation import Evaluation
from utils.evaluation import Index
from utils.loss import dice_loss
import cv2
import numpy as np
from tqdm import tqdm

alpha = 0.5
beta = 0.5


#
def train_net(net, device, data_path, val_path, test_path, log_dir, ModelName='BENet', epochs=150, batch_size=4,
              lr=0.0001, resume_from=None):
    # print(net)
    # Load dataset
    isbi_dataset = ISBI_Loader(data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=isbi_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
                                                                 120],
                                                     gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150],
    #                                                  gamma=0.9)

    # criterion = nn.BCEWithLogitsLoss()
    BCE_loss = nn.BCELoss()
    f_loss = open(log_dir + '/train_loss.txt', 'w')

    # f_time = open(log_dir + '/train_time.txt', 'w')

    start_epoch = 0
    if resume_from:
        # 加载预训练模型
        if os.path.exists(resume_from):
            print(f"Loading model from {resume_from}")
            net.load_state_dict(torch.load(resume_from, map_location=device))
            # 提取批次号
            start_epoch = int(os.path.basename(resume_from).split('_')[-1].split('.')[0])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Pretrained model file {resume_from} not found. Starting from scratch.")

    best_f1 = 0.

    for epoch in range(start_epoch + 1, epochs + 1):

        net.train()
        total_dice_loss, total_bce_loss, total_loss = 0, 0, 0
        num = int(0)

        # starttime = time.time()

        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch), colour='white') as t:
            for image, label in train_loader:
                optimizer.zero_grad()

                image = image.to(device=device)
                label = label.to(device=device)

                out = net(image)

                # compute loss
                dice_Loss = dice_loss(out, label)
                bce_Loss = BCE_loss(out, label)

                loss = alpha * dice_Loss + beta * bce_Loss
                # dice_Loss = dice_loss(out, label)
                # loss = criterion(pred2, label)

                if num == 0:
                    if epoch == 0:
                        f_loss.write('Note: epoch (num, epoch_loss)\n')
                        f_loss.write('epoch = ' + str(epoch) + '\n')
                    else:
                        f_loss.write('epoch = ' + str(epoch) + '\n')

                f_loss.write(str(num) + ',' + str(float('%5f' % loss)) + '\n')
                total_dice_loss += dice_Loss.item()
                total_bce_loss += bce_Loss.item()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                num += 1
                t.set_postfix({'lr': '%.8f' % optimizer.param_groups[0]['lr'],
                               'total_diss_loss': '%.4f' % (total_dice_loss / num),
                               'total_bce_loss': '%.4f' % (total_bce_loss / num),
                               'total_loss': '%.4f' % (total_loss / num),
                               })
                t.update(1)
        # learning rate delay
        scheduler.step()

        # endtime = time.time()
        # if epoch == 0:
        #     f_time.write('each epoch time\n')
        # f_time.write(str(epoch) + ',' + str(starttime) + ',' + str(endtime) + ',' + str(
        #     float('%4f' % (endtime - starttime))) + '\n')
        # val
        if epoch >= 0:
            with torch.no_grad():
                mOA, IoU, F1 = val(net, device, epoch, val_path)
                test(net, device, epoch, test_path)
                if F1 > best_f1:
                    best_f1 = F1
                    modelpath_best = 'BEST_' + str(ModelName) + '.pth'

                    torch.save(net.state_dict(), log_dir + "/pth/" + modelpath_best)
                # print(str(epoch) + ':::::OA=' + str(float('%2f' % (mOA))) + ':::::mIoU=' + str(float('%2f' % (IoU))))

                modelpath = 'last_' + str(ModelName) + '.pth'

                torch.save(net.state_dict(), log_dir + "/pth/" + modelpath)
    f_loss.close()
    # f_time.close()


def val(net, device, epoc, val_DataPath):
    net.eval()
    # image = glob.glob(os.path.join(val_DataPath, 'image/*.png'))
    # label = glob.glob(os.path.join(val_DataPath, 'label/*.png'))

    # 匹配 image 文件夹中的 .png 和 .tif 文件
    image_png = glob.glob(os.path.join(val_DataPath, 'image/*.png'))
    image_tif = glob.glob(os.path.join(val_DataPath, 'image/*.tif'))
    image = image_png + image_tif  # 合并两个列表

    # 匹配 label 文件夹中的 .png 和 .tif 文件
    label_png = glob.glob(os.path.join(val_DataPath, 'label/*.png'))
    label_tif = glob.glob(os.path.join(val_DataPath, 'label/*.tif'))
    label = label_png + label_tif  # 合并两个列表

    # # 匹配 boundary_png 文件夹中的 .png 和 .tif 文件
    # boundary_png = glob.glob(os.path.join(val_DataPath, 'boundary_png/*.png'))
    # boundary_tif = glob.glob(os.path.join(val_DataPath, 'boundary_png/*.tif'))
    # boundary = boundary_png + boundary_tif  # 合并两个列表

    trans = Transforms.Compose([Transforms.ToTensor()])
    BCE_loss = nn.BCELoss()

    IoU, c_IoU, uc_IoU, OA, Precision, Recall, F1 = 0, 0, 0, 0, 0, 0, 0
    num = 0
    total_loss_val = 0.
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0, 0, 0, 0, 0, 0
    val_acc = open(log_dir + '/val_all.txt', 'a')
    val_acc.write('===============================' + 'epoch=' + str(epoc) + '==============================\n')
    with tqdm(total=len(image), desc='Val Epoch #{}'.format(epoc), colour='yellow') as t:
        for val_path, label_path in zip(image, label):
            num += 1

            val_img = cv2.imread(val_path)
            val_label = cv2.imread(label_path)

            val_label = cv2.cvtColor(val_label, cv2.COLOR_BGR2GRAY)

            val_img = trans(val_img)
            val_img = val_img.unsqueeze(0)
            val_img = val_img.to(device=device, dtype=torch.float32)

            mask = trans(val_label).to(device=device, dtype=torch.float32)

            pred = net(val_img)

            dice_loss_val = dice_loss(pred, mask)
            bce_loss_val = BCE_loss(pred, mask.unsqueeze(1))

            loss_val = alpha * dice_loss_val + beta * bce_loss_val

            total_loss_val += loss_val.item()

            # acquire result
            pred = np.array(pred.data.cpu()[0])[0]
            # binary map
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            monfusion_matrix = Evaluation(label=val_label, pred=pred)
            TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
            TPSum += TP
            TNSum += TN
            FPSum += FP
            FNSum += FN
            C_Sum_or += c_num_or
            UC_Sum_or += uc_num_or

            if num > 0:
                Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
                IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
                OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()

            t.set_postfix({
                'total_loss_val': '%.4f' % (total_loss_val / num),
                'mIoU': '%.4f' % IoU,
                'c_IoU': '%.4f' % c_IoU,
                'uc_IoU': '%.4f' % uc_IoU,
                'OA': '%.4f' % OA,
                'PRE': '%.4f' % Precision,
                'REC': '%.4f' % Recall,
                'F1': '%.4f' % F1})
            t.update(1)
    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    val_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                  str(float('%2f' % (c_IoU))) + ',' +
                  'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                  'OA = ' + str(float('%2f' % (OA))) + ',' +
                  'PRE = ' + str(float('%2f' % (Precision))) + ',' +
                  'REC = ' + str(float('%2f' % (Recall))) + ',' +
                  'F1 = ' + str(float('%2f' % (F1))) + '\n')
    val_acc.close()
    return OA, IoU, F1


def test(net, device, epoc, test_DataPath):
    net.eval()
    # image = glob.glob(os.path.join(val_DataPath, 'image/*.png'))
    # label = glob.glob(os.path.join(val_DataPath, 'label/*.png'))

    # 匹配 image 文件夹中的 .png 和 .tif 文件
    image_png = glob.glob(os.path.join(test_DataPath, 'image/*.png'))
    image_tif = glob.glob(os.path.join(test_DataPath, 'image/*.tif'))
    image = image_png + image_tif  # 合并两个列表

    # 匹配 label 文件夹中的 .png 和 .tif 文件
    label_png = glob.glob(os.path.join(test_DataPath, 'label/*.png'))
    label_tif = glob.glob(os.path.join(test_DataPath, 'label/*.tif'))
    label = label_png + label_tif  # 合并两个列表

    trans = Transforms.Compose([Transforms.ToTensor()])
    # BCE_loss = nn.BCELoss()

    IoU, c_IoU, uc_IoU, OA, Precision, Recall, F1 = 0, 0, 0, 0, 0, 0, 0
    num = 0
    # total_loss_val = 0.
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0, 0, 0, 0, 0, 0
    val_acc = open(log_dir + '/test_all.txt', 'a')
    val_acc.write('===============================' + 'epoch=' + str(epoc) + '==============================\n')
    with tqdm(total=len(image), desc='Test Epoch #{}'.format(epoc), colour='blue') as t:
        for val_path, label_path in zip(image, label):
            num += 1

            val_img = cv2.imread(val_path)
            val_label = cv2.imread(label_path)

            val_label = cv2.cvtColor(val_label, cv2.COLOR_BGR2GRAY)

            val_img = trans(val_img)
            val_img = val_img.unsqueeze(0)
            val_img = val_img.to(device=device, dtype=torch.float32)

            pred = net(val_img)

            # dice_loss_val = dice_loss(pred, mask)
            # bce_loss_val = BCE_loss(pred, mask.unsqueeze(1))

            # loss_val = alpha * dice_loss_val + beta * bce_loss_val

            # total_loss_val += loss_val.item()

            # acquire result
            pred = np.array(pred.data.cpu()[0])[0]
            # binary map
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            monfusion_matrix = Evaluation(label=val_label, pred=pred)
            TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
            TPSum += TP
            TNSum += TN
            FPSum += FP
            FNSum += FN
            C_Sum_or += c_num_or
            UC_Sum_or += uc_num_or

            if num > 0:
                Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
                IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
                OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()

            t.set_postfix({
                # 'total_loss_val': '%.4f' % (total_loss_val / num),
                'mIoU': '%.4f' % IoU,
                'c_IoU': '%.4f' % c_IoU,
                'uc_IoU': '%.4f' % uc_IoU,
                'OA': '%.4f' % OA,
                'PRE': '%.4f' % Precision,
                'REC': '%.4f' % Recall,
                'F1': '%.4f' % F1})
            t.update(1)
    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    val_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                  str(float('%2f' % (c_IoU))) + ',' +
                  'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                  'OA = ' + str(float('%2f' % (OA))) + ',' +
                  'PRE = ' + str(float('%2f' % (Precision))) + ',' +
                  'REC = ' + str(float('%2f' % (Recall))) + ',' +
                  'F1 = ' + str(float('%2f' % (F1))) + '\n')
    val_acc.close()
    return OA, IoU, F1


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = BENet()
    net.to(device=device)
    log_dir = "result/mass"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train_path = "data/mass/mass256/train"
    val_path = "data/mass/mass256/val"
    test_path = "data/mass/mass256/test"

    # 指定从哪个预训练模型继续训练
    resume_from = None

    train_net(net, device, train_path, test_path, test_path, log_dir, batch_size=4, epochs=100, lr=0.0001,
              resume_from=resume_from)

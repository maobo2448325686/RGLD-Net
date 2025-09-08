import torchvision.transforms as Transforms
import torch
import glob
import os

from torch import nn
from ptflops import get_model_complexity_info
from model.BENet import BENet
from utils.evaluation import Evaluation
from utils.evaluation import Index
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

log_dir = "result/whu"
image_pre_dir = "result/whu/image_pre"

def count_parameters(model):
    """
    计算模型的参数量，并换算为百万（M）为单位
    :param model: PyTorch 模型
    :return: 模型的总参数量、可训练参数量和非可训练参数量（单位：M）
    """
    total_params = sum(p.numel() for p in model.parameters())  # 计算总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 计算可训练参数量
    non_trainable_params = total_params - trainable_params  # 计算非可训练参数量

    # 换算为百万（M）为单位
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    non_trainable_params_m = non_trainable_params / 1e6

    return total_params_m, trainable_params_m, non_trainable_params_m


def img(pr, gt, filename):
    pr = (pr > 0.5).float()
    pr = pr[0, 0].cpu().detach().numpy()  # 转换为 NumPy 数组

    # 确保 gt 的值是二值的（0 或 1）
    gt = (gt > 0.5).astype(np.uint8)

    tp = np.logical_and(pr == 1, gt == 1)
    fp = np.logical_and(pr == 1, gt == 0)
    tn = np.logical_and(pr == 0, gt == 0)
    fn = np.logical_and(pr == 0, gt == 1)

    image = np.zeros([256, 256, 3], dtype=np.uint8)

    image[tp] = [255, 255, 255]  # 真正例 (白色)
    image[fp] = [255, 0, 0]      # 假正例 (红色)
    image[tn] = [0, 0, 0]        # 真负例 (黑色)
    image[fn] = [0, 255, 0]      # 假负例 (绿色)

    res_image = Image.fromarray(image)
    save_path = os.path.join(image_pre_dir, filename)
    # print()
    res_image.save(save_path)


#
def test_net(net, device, test_path, ModelName='BENet', epochs=100):
    for epoch in range(epochs):
        # test
        if epoch >= 0:
            test(net, device, epoch, test_path)


def test(net, device, epoch, test_DataPath):

    # 计算模型参数量大小
    # total_params, trainable_params, non_trainable_params = count_parameters(net)
    # print(f"Trainable parameters: {trainable_params} M")


    # 加载特定 epoch 的模型参数
    model_path = os.path.join(log_dir, f"BEST_BENet.pth")
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("加载成功！")
    else:
        print(f"Model file not found: {model_path}")

    net.eval()


    # 计算参数量和flops
    macs, params = get_model_complexity_info(
        net,
        input_res=(3, 256, 256),  # 两个 3 通道图像拼接在一起
        as_strings=False,
        print_per_layer_stat=False
    )
    # 转换为 G 和 M
    flops_G = macs * 0.5 / 1e9
    params_M = params / 1e6
    print(f'Params: {params_M:.2f} M, FLOPs: {flops_G:.2f} G')


    # 匹配 image 文件夹中的 .png 和 .tif 文件
    image_png = glob.glob(os.path.join(test_DataPath, 'image/*.png'))
    image_tif = glob.glob(os.path.join(test_DataPath, 'image/*.tif'))
    image = image_png + image_tif  # 合并两个列表

    # 匹配 label 文件夹中的 .png 和 .tif 文件
    label_png = glob.glob(os.path.join(test_DataPath, 'label/*.png'))
    label_tif = glob.glob(os.path.join(test_DataPath, 'label/*.tif'))
    label = label_png + label_tif  # 合并两个列表



    trans = Transforms.Compose([Transforms.ToTensor()])
    IoU, c_IoU, uc_IoU, OA, Precision, Recall, F1 = 0., 0., 0., 0., 0., 0., 0.
    num = 0
    TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or = 0., 0., 0., 0., 0., 0.
    test_acc = open(log_dir + '/test_only.txt', 'a')
    test_acc.write('total_params' + str(params_M) + ' M\n')
    test_acc.write('flops_G' + str(flops_G) + ' M\n')
    test_acc.write('===============================' + 'epoch=' + str(epoch) + '==============================\n')
    with tqdm(total=len(image), desc='test Epoch #{}'.format(epoch), colour='blue') as t:
        for test_path, label_path in zip(image, label):
            num += 1
            filename = os.path.basename(test_path)
            test_img = cv2.imread(test_path)
            test_label_old = cv2.imread(label_path)
            test_label = cv2.cvtColor(test_label_old, cv2.COLOR_BGR2GRAY)
            test_img = trans(test_img)

            test_img = test_img.unsqueeze(0)

            test_img = test_img.to(device=device, dtype=torch.float32)

            pred = net(test_img)

            # image_pre
            # img(pred, test_label, filename)

            # acquire result
            pred = np.array(pred.data.cpu()[0])[0]
            # binary map
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0

            monfusion_matrix = Evaluation(label=test_label, pred=pred)
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
                'OA': '%.4f' % OA,
                'mIoU': '%.4f' % IoU,
                'c_IoU': '%.4f' % c_IoU,
                'uc_IoU': '%.4f' % uc_IoU,
                'PRE': '%.4f' % Precision,
                'REC': '%.4f' % Recall,
                'F1': '%.4f' % F1})
            t.update(1)
    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    test_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                   str(float('%2f' % (c_IoU))) + ',' +
                   'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                   'OA = ' + str(float('%2f' % (OA))) + ',' +
                   'PRE = ' + str(float('%2f' % (Precision))) + ',' +
                   'REC = ' + str(float('%2f' % (Recall))) + ',' +
                   'F1 = ' + str(float('%2f' % (F1))) + '\n')
    test_acc.close()
    return OA, IoU


if __name__ == '__main__':

    net = BENet()
    net.to(device=device)
    epochs = 100
    test_path = r"data/mass/mass256/test"
    epoch = 0
    test(net, device, epoch=epoch, test_DataPath=test_path)
    # for epoch in range(epochs):
    #     test(net, device, epoch, test_path)

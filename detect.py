import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from od.data.datasets import LoadImages, create_dataloader
from od.models.modules import attempt_load
from utils.evaluation_utils import rbox2txt
from utils.general import (
    check_img_size, apply_classifier, scale_labels,
    xyxy2xywh, strip_optimizer, set_logging, rotate_non_max_suppression, colorstr, increment_path)
from utils.plots import plot_images, output_to_target, plot_one_rotated_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tqdm import tqdm


# def detect(save_img=False):
def detect(opt,
           weights=None,
           model=None,
           dataloader=None,
           compute_loss=None,
           batch_size=1,
           save_dir=Path(''),
           plots=False,  # plots detect result
           visualize=False,  # visualize features
           log_imgs=0,  # whether to upload test images (upload per 5 epochs)
           ):
    # 获取输出文件夹，输入路径，权重，参数等参数
    out, small_datasets, save_txt, imgsz = \
        opt.detect_output, opt.small_datasets, opt.save_txt, opt.img_size[0]

    if opt.small_datasets:
        source = '../datasets/dota_interest_small/images/val'
    elif plots:  # test and plots
        source = '../datasets/dota_interest_{}/images/test'.format(imgsz)
    else:
        source = '../datasets/dota_interest_{}/images/val'.format(imgsz)

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:
        set_logging()
        device = select_device(opt.device)
        # 加载Float32模型，确保用户设定的输入图片分辨率能整除最大步长s=32(如不能则调整为能整除并返回)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        # test path
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)

    # 移除之前的输出文件夹,并新建输出文件夹
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half and not visualize:
        model.half()  # 设置Float16
    model.eval()

    try:
        import wandb  # Weights & Biases
    except ImportError:
        wandb = None

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    if not training:
        # dataloader = LoadImages(source, img_size=imgsz)  # val
        dataloader = create_dataloader(source, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=False,
                                       prefix=colorstr('test: ' if opt.task == 'test' else 'val: '))[0]

    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference 进行一次前向推理,测试程序是否正常  向量维度（1，3，imgsz，imgsz）
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0, t1 = 0., 0.
    seen = 0
    if compute_loss:
        s = ('%20s' + '%10.4g' * 4 + '%20s' + '%10s') % ('', 0, 0, 0, 0, '', 'detect')
    else:
        s = ('%10s' % 'detect')
    loss = torch.zeros(4, device=device)
    # 设置画框的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    pbar = tqdm(dataloader, desc=s)
    for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
        # img = torch.from_numpy(img).to(device)
        img = img.to(device, non_blocking=True)
        # 图片也设置为Float16
        img = img.half() if half and not visualize else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        if visualize:  # visualize features
            visualize = increment_path(save_dir / Path(paths[0]).stem)
            Path(visualize).mkdir(parents=True, exist_ok=True)

        # inference
        with torch.no_grad():
            # Inference
            t = time_synchronized()
            # pred -> (batch_size, boxes, cls)  -> batch_size=1, cls = 16 + 5 + 180
            pred, train_out = model(img, visualize=visualize)
            t0 += time_synchronized() - t  # inference time

            # 计算验证集损失
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:4]  # box, obj, cls, angle obb

            # Apply NMS
            # pred : list[tensor(batch_size, num_conf_nms, [xylsθ,conf,classid])] θ∈[0,179]
            # pred => xywhθ, conf, classid
            # iou_thres=0.45
            t = time_synchronized()
            pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                          agnostic=opt.agnostic_nms, without_iouthres=False)
            t1 += time_synchronized() - t  # nms time

        # Process detections
        for i, det in enumerate(pred):  # i:image index  det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
            p, s = paths[i], ''
            im0_name = p.split('/')[-1]
            im0 = cv2.imread(p)
            im0_shape = shapes[0][0]
            seen += 1
            # save_path = str(Path(out) / Path(p).name)  # 图片保存路径+图片名字
            # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataloader.frame if dataloader.mode == 'video' else '')
            # print(txt_path)
            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0_shape).round()

                # Print results    det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
                for c in det[:, -1].unique():  # unique函数去除其中重复的元素，并按元素（类别）由大到小返回一个新的无元素重复的元组或者列表
                    n = (det[:, -1] == c).sum()  # detections per class  每个类别检测出来的素含量
                    # s += '%g %ss, ' % (n, names[int(c)])  # add to string 输出‘数量 类别,’

                # Write results  det:(num_nms_boxes, [xywhθ,conf,classid]) θ∈[0,179]
                for *rbox, conf, cls in reversed(det):  # 翻转list的排列结果,改为类别由小到大的排列
                    # rbox=[tensor(x),tensor(y),tensor(w),tensor(h),tsneor(θ)] θ∈[0,179]
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    if plots:
                        label = '%d' % (int(cls))
                    else:
                        label = '%s %.2f' % (names[int(cls)], conf)
                    classname = '%s' % names[int(cls)]
                    conf_str = '%.3f' % conf
                    # write detection result txt before merge
                    rbox2txt(rbox, classname, conf_str, Path(p).stem, str(out + '/result_txt/result_before_merge'))
                    if plots or batch_i < 3:
                        # Add bbox to image
                        plot_one_rotated_box(rbox, im0, label=label, color=colors[int(cls)], line_thickness=1)
            # Save results (image with detections)
            if plots or batch_i < 3:
                cv2.imwrite(str(save_dir/('test_'+im0_name)), im0)
            elif batch_i == 10 and wandb and log_imgs:
                wandb.log(
                    {"Validation": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('test*.png')
                                    if x.exists()]},
                    commit=False)


        # show val loss
        pred_count = len(pred[0]) if pred[0] is not None else 0
        if training:
            lbox, lobj, lcls, langle = loss / ((batch_i + 1) * batch_size)
            ltotal = lbox + lobj + lcls + langle
            s = ('%20s' + '%10.4g' * 5 + '%10d' + '%10s') % ('', lbox, lobj, lcls, langle, ltotal, pred_count, 'detect')
        else:
            s = '{:^10s} {:^10d}'.format('pred:', pred_count)

        pbar.set_description(s)

    # Print time (inference + NMS)
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    if save_txt or plots:
        print('   Results saved to %s' % Path(save_dir))

    model.float()  # for training
    return (loss.cpu() / len(dataloader)).tolist()


if __name__ == '__main__':
    """
        weights:训练的权重
        source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
        output:网络预测之后的图片/视频的保存路径
        img-size:网络输入图片大小
        conf-thres:置信度阈值
        iou-thres:做nms的iou阈值
        device:设置设备
        view-img:是否展示预测之后的图片/视频，默认False
        save-txt:是否将预测的框坐标以txt文件形式保存，默认False
        classes:设置只保留某一部分类别，形如0或者0 2 3
        agnostic-nms:进行nms是否将所有类别框一视同仁，默认False
        augment:推理的时候进行多尺度，翻转等操作(TTA)推理
        update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp-swinS-p2-68-map68/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--detect_output', type=str, default='DOTA/detection', help='output folder')  # output folder
    parser.add_argument('--small-datasets', action='store_true', help='display results')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1024, 1024], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plots', action='store_true', help='plots when not training')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt, weights=opt.weights, plots=opt.plots, visualize=opt.visualize)

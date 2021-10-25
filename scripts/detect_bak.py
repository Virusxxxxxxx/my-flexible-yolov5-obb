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

from od.data.datasets import LoadImages
from od.models.modules import attempt_load
from utils.evaluation_utils import rbox2txt
from utils.general import (
    check_img_size, apply_classifier, scale_labels,
    xyxy2xywh, strip_optimizer, set_logging, rotate_non_max_suppression)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tqdm import tqdm


# def detect(save_img=False):
def detect(opt, weights=None, model=None, save_img=False):
    # 获取输出文件夹，输入路径，权重，参数等参数
    out, small_datasets, view_img, save_txt, imgsz, save_dir = \
        opt.detect_output, opt.small_datasets, opt.view_img, opt.save_txt, opt.img_size[0], opt.save_dir

    source = '../datasets/dota_interest_small/images/val' if opt.small_datasets else '../datasets/dota_interest/images/val'

    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:
        device = select_device(opt.device)
        # 加载Float32模型，确保用户设定的输入图片分辨率能整除最大步长s=32(如不能则调整为能整除并返回)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Initialize
    set_logging()
    # 移除之前的输出文件夹,并新建输出文件夹
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # 设置Float16
    model.eval()

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz)
    # else:
    #     save_img = True
    #     dataset = LoadImages(source, img_size=imgsz)  # val

    save_img = True
    dataset = LoadImages(source, img_size=imgsz)  # val

    # Get names and colors
    # 获取类别名字    names = ['person', 'bicycle', 'car',...,'toothbrush']
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # 进行一次前向推理,测试程序是否正常  向量维度（1，3，imgsz，imgsz）
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    pbar = enumerate(dataset)
    pbar = tqdm(pbar, total=len(dataset))

    for i, (path, img, im0s, vid_cap) in pbar:
        # print(img.shape)
        img = torch.from_numpy(img).to(device)
        # 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            # (in_channels,size1,size2) to (1,in_channels,img_height,img_weight)
            img = img.unsqueeze(0)  # 在[0]维增加一个维度

        with torch.no_grad():
            # Inference
            t1 = time_synchronized()
            # pred : (batch_size, boxes, cls)  batch_size=1, cls = 16 + 5 + 180
            pred, train_out = model(img)

            # loss, loss_items = compute_loss(train_out, targets.to(device), model,
            #                                 csl_label_flag=True)  # loss scaled by batch_size
            # TODO 计算验证集损失
            # loss += compute_loss([x.float() for x in train_out], targets)[1][:4]  # box, obj, cls, angle obb

            # Apply NMS
            # 进行NMS
            # pred : list[tensor(batch_size, num_conf_nms, [xylsθ,conf,classid])] θ∈[0,179]
            # pred => xywhθ, conf, classid
            # iou_thres=0.45
            pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                          agnostic=opt.agnostic_nms, without_iouthres=False)
            t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # i:image index  det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)  # 图片保存路径+图片名字
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            # print(txt_path)
            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()

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

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        classname = '%s' % names[int(cls)]
                        conf_str = '%.3f' % conf
                        rbox2txt(rbox, classname, conf_str, Path(p).stem, str(out + '/result_txt/result_before_merge'))
                        # plot_one_box(rbox, im0, label=label, color=colors[int(cls)], line_thickness=2)

                        # TODO 画detect效果图，再整合wandb
                        # if i < 3:
                        #     f = save_dir / f'test_batch{i}_labels.jpg'  # labels
                        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
                        #     f = save_dir / f'test_batch{i}_pred.jpg'  # predictions
                        #     Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names),
                        #            daemon=True).start()

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results 播放结果
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            # save_img = False
            # if save_img:
            #     if dataset.mode == 'images':
            #         cv2.imwrite(save_path, im0)
            #         pass
            #     else:
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #
            #             fourcc = 'mp4v'  # output video codec
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            #         vid_writer.write(im0)
        s = '%90s' % 'detect'
        pbar.set_description(s)

    model.float()  # for training
    # if save_txt or save_img:
    # print('   Results saved to %s' % Path(out))
    # print('   Detection Done. (%.3f s)' % (time.time() - t0))


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
    parser.add_argument('--weights', nargs='+', type=str, default='',
                        help='model.pt path(s)')
    parser.add_argument('--detect_source', type=str, default='DOTA_demo_view/images/val',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--detect_output', type=str, default='DOTA_demo_view/detection', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=[1024, 1024], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                # 去除pt文件中的优化器等信息
                strip_optimizer(opt.weights)
        else:
            detect(opt, weights='runs/exp19/weights/yolov5m_p4.pt')

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync
from utils.augmentations import letterbox
from vector.contour import bit_detection
from vector.homography import homography, findcorners


# Load model
imgsz=(416, 416)
device = select_device(0)

model = DetectMultiBackend('vector/graphs/object-detector.pt', device=device, dnn=False, data='vector/graphs/object.yaml', fp16=True)
stride, names, pt = model.stride, model.names, model.pt
imgsz1 = check_img_size(imgsz, s=stride)  # check image size
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz1))  # warmup

model2 = DetectMultiBackend('vector/graphs/panel-detector.pt', device=device, dnn=False, data='vector/graphs/panel.yaml', fp16=True)
imgsz2 = check_img_size(imgsz, s=stride)  # check image size
stride2, names2, pt2 = model2.stride, model2.names, model2.pt
model2.warmup(imgsz=(1 if pt else bs, 3, *imgsz2))  # warmup

#source = '1'
source='vector/panel-test.mp4'
if source =='1':
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)  # batch_size
else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1

vid_path, vid_writer = [None] * 1, [None] * 1
save_dir = save_dir = increment_path(Path('data/'), exist_ok=True)

names = ['pedestrian', 'cyclist', 'vehicle-car', 'vehicle-bus', 'vehicle-truck', 'train', 'traffic light', 'traffic sign']
for path, im, im0s, vid_cap, s in dataset:
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()
    im = im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.35, 0.45, None, False, max_det=200)

    for i, det in enumerate(pred):
        if source == '1':
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '
        else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=3, example=str(names))
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        imc = im0.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):

                
                c = int(cls)  # integer class
                label=(f'{names[c]} {conf:.2f}')
                if c == 2:
                    xyxy[1] -= (xyxy[3]-xyxy[1])*0.1
                    color=colors(10, True)         
                else:
                    color=colors(5, True)   

                #imv = save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)     
                imv0 = save_one_box(xyxy, imc, BGR=True, save=False) 
                annotator.box_label(xyxy, label, color=color)

                # Detect Axles and Plates
                imv = letterbox(imv0, imgsz2, stride=stride2, auto=True)[0]  # padded resize
                imv = imv.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                imv = np.ascontiguousarray(imv)  # contiguous
                imv = torch.from_numpy(imv).to(device)
                imv = imv.float()
                imv /= 255
                if len(imv.shape) == 3:
                    imv = imv[None]  # expand for batch dim
                pred2 = model2(imv, augment=False, visualize=False)
                pred2 = non_max_suppression(pred2, 0.45, 0.45, None, False, max_det=100)
                
                for j, det2 in enumerate(pred2): 
                    gn_ = torch.tensor(imv.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det2[:, :4] = scale_coords(imv.shape[2:], det2[:, :4], imv0.shape).round()
                        for *xyxy2, conf2, cls2 in reversed(det2):
                            wp = xyxy2[2] - xyxy2[0]
                            hp = xyxy2[3] - xyxy2[1]
                            xyxy2[0] += xyxy[0] - wp*0.1
                            xyxy2[1] += xyxy[1] - hp*0.1
                            xyxy2[2] += xyxy[0] + hp*0.1
                            xyxy2[3] += xyxy[1] + hp*0.1
                            #annotator.box_label(xyxy2, label, color=colors(0, True))
                            if int(cls2) == 0:
                                label2=(f'V2X message {conf2:.2f}') 
                                annotator.box_label(xyxy2, label2, color=colors(0, True))
                                msg_img = save_one_box(xyxy2, imc, BGR=True, save=False)
                                msg_img = cv2.resize(msg_img, (1920,320))

                                #corners = findcorners(msg_img)
                                #msg_img = homography(msg_img)
                                code, check, message = bit_detection(msg_img)
        
               
        im_result = annotator.result()
        im_result = cv2.resize(im_result, (1920, 1080))
        im_result[760:1080,:,:]=message
        im_result[0:250,0:600,:]=35
        cv2.putText(im_result, "Decoded V2X Message", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.imshow(str(p), im_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        


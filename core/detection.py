from typing import List, Tuple, Optional
import numpy as np
import cv2 as cv
import torch
import time
from backbones.torchRetina import TorchRetina
from components.functions import decode, decode_landm, PriorBox, py_cpu_nms
from backbones.net_config import cfg_mnet

class RetinaDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = TorchRetina(backbone_name='mnet25', cfg=cfg_mnet)
        self.net.load_state_dict(torch.load('weights/mobilenet0.25_Final.pth', map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        
        self.resize = 1
        self.conf_thresh = 0.5
        self.nms_thresh = 0.4
        self.vis_thresh = 0.6
        self.p_top_k = 5000
        self.p_keep_top_k = 750
        print('Finished loading retina model!')
        
    def detect(self, frame=None):
        if frame is None:
            return [], []
        img = np.float32(frame)

        # processing
        im_height, im_width = frame.shape[:2]
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.to(self.device)
        scale = scale.to(self.device)

        # forwarding
        loc, conf, landms = self.net(img)
        
        pBox = PriorBox(image_size=(im_height, im_width))
        priors = pBox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.p_top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, self.nms_thresh)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.p_keep_top_k, :]
        landms = landms[:self.p_keep_top_k, :]

        faces = []
        ret_landms = []
        
        for idx in range(len(dets)):
            bbox = dets[idx]
            if bbox[4] < self.vis_thresh:
                continue
            bbox = list(map(int, bbox))

            x1 = bbox[0]-20 if bbox[0]-20 >= 0 else bbox[0]
            y1 = bbox[1]-20 if bbox[1]-20 >= 0 else bbox[1]
            x2 = bbox[2]+20 if bbox[2]+20 >= 0 else bbox[2]
            y2 = bbox[3]+20 if bbox[3]+20 >= 0 else bbox[3]
            faces.append([x1, y1, x2, y2, bbox[4]])

            landm = landms[idx]
            i_width = bbox[2] - bbox[0]
            i_height = bbox[3] - bbox[1]
            b_start_x = bbox[0]
            b_start_y = bbox[1]
            ret_landm = []
            
            for land_idx in range(0, len(landm), 2):
                land_x = landm[land_idx]
                land_y = landm[land_idx + 1]
                land_w = (land_x - b_start_x) / i_width
                land_h = (land_y - b_start_y) / i_height
                ret_landm.append(land_w)
                ret_landm.append(land_h)
            ret_landms.append(ret_landm)
                
        return faces, ret_landms

class CascadeDetector:
    def __init__(self):
        self.face_cascade = cv.CascadeClassifier('weights/haarcascade_profileface.xml')
        
    def detect(self, frame=None):
        if frame is None:
            return [], []
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.3, 4)
        ret_landms = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in faces]  # Empty landmarks for cascade detector
        return faces, ret_landms

def detect(self, frame=None):
    if frame is None:
        return [], []
    img = np.float32(frame)

    # processing
    im_height, im_width = frame.shape[:2]
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    
    # FIX: Ensure the tensor is FloatTensor, not DoubleTensor
    img = torch.from_numpy(img).unsqueeze(0).float()  # Added .float()
    img = img.to(self.device)
    scale = scale.to(self.device)

    # forwarding
    t = time.time()
    loc, conf, landms = self.net(img)
    # print(f'net forward time: {time.time() - t}')
    
    pBox = PriorBox(image_size=(im_height, im_width))
    priors = pBox.forward()
    priors = priors.to(self.device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
    boxes = boxes * scale / self.resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(self.device)
    landms = landms * scale1 / self.resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > self.conf_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:self.p_top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = py_cpu_nms(dets, self.nms_thresh)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:self.p_keep_top_k, :]
    landms = landms[:self.p_keep_top_k, :]

    # process the returned result for recognizer
    # extract faces
    faces = []
    # extract landmarks
    ret_landms = []
    
    for idx in range(len(dets)):
        bbox = dets[idx]
        # skip unconf boxes
        if bbox[4] < self.vis_thresh:
            continue
        bbox = list(map(int, bbox))

        # getting bbox with padding
        x1 = bbox[0]-20 if bbox[0]-20 >= 0 else bbox[0]
        y1 = bbox[1]-20 if bbox[1]-20 >= 0 else bbox[1]
        x2 = bbox[2]+20 if bbox[2]+20 >= 0 else bbox[2]
        y2 = bbox[3]+20 if bbox[3]+20 >= 0 else bbox[3]
        faces.append([x1, y1, x2, y2, bbox[4]])  # x1 y1 x2 y2 confidence

        landm = landms[idx]
        i_width = bbox[2] - bbox[0]
        i_height = bbox[3] - bbox[1]
        b_start_x = bbox[0]
        b_start_y = bbox[1]
        ret_landm = []
        # change landmark points to scale with w,h of bbox
        for land_idx in range(0, len(landm), 2):
            land_x = landm[land_idx]
            land_y = landm[land_idx + 1]

            land_w = (land_x - b_start_x) / i_width
            land_h = (land_y - b_start_y) / i_height
            ret_landm.append(land_w)
            ret_landm.append(land_h)
        ret_landms.append(ret_landm)
            
    return faces, ret_landms
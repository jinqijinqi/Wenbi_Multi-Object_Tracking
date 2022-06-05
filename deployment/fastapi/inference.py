import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append("./src/lib")

from src.lib.tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.timer import Timer
import datasets.dataset.jde as datasets

from opts import opts


class OD:
    def __init__(self,img_size=(960, 576)):
        self.opt=opts().init()
        self.opt.load_model=r"D:\毕业论文材料\PedestrianDetectionSystem-master\python\models\fairmot_lite.pth"
        self.opt.arch=r"yolo"
        self.opt.reid_dim = 64
        self.opt.heads ={'hm': 1, 'wh': 4, 'id': 64, 'reg': 2}
        self.width = img_size[0]
        self.height = img_size[1]
        self.tracker=JDETracker(self.opt, frame_rate=30)

    def infer(self, img0 ,use_cuda=True):
        assert img0 is not None, 'Failed to load '

        # Padded resize
        img, _, _, _ = datasets.letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        timer = Timer()
        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets =self.tracker.update(blob.half(), img0)
        online_tlwhs = []
        online_ids = []
        # online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                # online_scores.append(t.score)
        timer.toc()
        print(len(online_tlwhs))
        online_im = vis.plot_tracking(img0, online_tlwhs, online_ids,
                                      fps=1. / timer.average_time)
        print(len(online_tlwhs))
        return online_im,len(online_tlwhs)

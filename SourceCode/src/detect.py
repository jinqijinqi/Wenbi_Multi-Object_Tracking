from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch.nn.functional as F

from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
import torch
from tracking_utils.utils import mkdir_if_missing
from opts import opts
from models.decode import mot_decode
from utils.post_process import ctdet_post_process
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat

num_classes = 1
max_per_image = 500
reid_dim = 128
ltrb = True
reg_offset = True
conf_thres = 0.2
Kt = 500
heads = {'hm': num_classes, 'wh': 2 if not ltrb else 4, 'id': reid_dim, 'reg': 2}
head_conv = 256
down_ratio = 4
def write_results_score(filename, results):
    save_format = '{frame},{x1},{y1},{w},{h},{s}\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, scores in results:
            for tlwh, score in zip(tlwhs, scores):
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, x1=x1, y1=y1, w=w, h=h, s=score)
                f.write(line)
    print('save results to {}'.format(filename))


def post_process(dets, meta):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    #print('dets',dets[0].keys())
    return dets[0]


def merge_outputs(detections):
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)
    model.eval()
    timer = Timer()
    results = []
    len_all = len(dataloader)
    start_frame = int(len_all / 2)
    frame_id = int(len_all / 2)
    for i, (path, img, img0) in enumerate(dataloader):
        if i < start_frame:
            continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # run detecting
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = blob.shape[2]
        inp_width = blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // opt.down_ratio,
                'out_width': inp_width // opt.down_ratio}
        with torch.no_grad():
            output = model(blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg']
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=True, K=500)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = post_process(dets,meta)
        dets = merge_outputs([dets])[1]
        remain_inds = dets[:, 4] > conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # tlwhs = []
        # scores = []
        # for *tlwh, conf in dets:
        #     tlwhs.append(tlwh)
        #     scores.append(conf)
        timer.toc()
        line_thickness = max(1, int(img0.shape[1] / 500.))
        text_scale = max(1, img0.shape[1] / 1600.)
        text_thickness = 2
        # save results
        # results.append((frame_id + 1, tlwhs, scores))
        if show_image or save_dir is not None:
            for i in range(0, dets.shape[0]):
                bbox = dets[i][0:4]
                cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(img0, str(round(dets[i][4],2)) , (int(bbox[0]), int(bbox[1]) + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                            thickness=text_thickness)
        if show_image:
            cv2.imshow('online_im', img0)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), img0)
        frame_id += 1
    # save results
    # write_results_score(result_filename, results)
    #write_results_score_hie(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls

def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:4] -= ret[:2]
    return ret

def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'dets', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))

        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    if opt.val_hie:
        seqs_str = '''1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19'''
        #seqs_str = '''9'''
        #seqs_str = '''11 12 13 14 15 16 17 18 19'''
        data_root = '/data/yfzhang/MOT/JDE/HIE/HIE20/images/train'
    elif opt.test_hie:
        seqs_str = '''20 21 22 23 24 25 26 27 28 29 30 31 32'''
        seqs_str = '''25'''
        data_root = '/data/yfzhang/MOT/JDE/HIE/HIE20/images/test'
    elif opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        #seqs_str = '''MOT17-02-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    else:
        seqs_str = None
        data_root = None
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='dla34_giou_fairmot_mot17',
         show_image=False,
         save_images=True,
         save_videos=False)

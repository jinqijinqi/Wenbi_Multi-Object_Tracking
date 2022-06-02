import io

import requests
from PIL import Image
import src._init_paths
src._init_paths.add_path('./src')

import streamlit as st
from src.lib.opts import opts
from src.demo import demo
from io import StringIO
from pathlib import Path
import time
import os
import sys
from PIL import Image
import argparse

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\track
    '''
    return max(get_subdirs(os.path.join('runs', 'track')), key=os.path.getmtime)


if __name__ == '__main__':

    st.title('MOT Streamlit App')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    source = ('目标跟踪视频',)
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='资源加载中...'):
            st.sidebar.video(uploaded_file)
            with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            opt.input_video = f'data/videos/{uploaded_file.name}'
    else:
        is_valid = False

    opt.output_root = os.path.join('runs', 'track')
    opt.arch = "mit_1"
    opt.load_model = os.path.join('models','model_20.pth')
    opt.conf_thres = 0.4

    if is_valid:
        print('valid')
        if st.button('开始检测跟踪'):
            # video_file = open(r'D:\streamlit-fastapi-model-serving\fastapi\data\videos\5月27日.mp4', 'rb')
            # print(video_file)
            # video_bytes = video_file.read()
            # st.video(video_bytes, format='video/mp4', start_time=0)
            demo(opt)
            with st.spinner(text='Preparing Video'):
                video_file = open('./runs/track/results.mp4', 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes, format='video/mp4', start_time=0)
                st.balloons()
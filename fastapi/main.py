

import cv2
import numpy as np
import requests

from inference import OD
import subprocess as sp
import threading
import time
from PIL import Image
import json
from flask import Flask, jsonify,request,make_response
app=Flask(__name__)
RTMP_HOST=r"rtmp://127.0.0.1:1935/live/1"
rtmpUrl = r"rtmp://127.0.0.1:1935/live/123"

od = OD()
use_channel = 1
shared_image = (np.ones((576, 960, 3), dtype=np.uint8) * 255).astype(np.uint8)
process_image = (np.ones((576, 960, 3), dtype=np.uint8) * 255).astype(np.uint8)
people_count = 2
count=0

@app.route('/peoplenum',methods=['GET','POST'])
def peoplenum():
    global people_count
    method = request.method
    res = make_response(jsonify(people_count=people_count,method=method))  # 设置响应体
    res.status = '200'  # 设置状态码
    res.status = '200'  # 设置状态码
    res.headers['Access-Control-Allow-Origin'] = "*"  # 设置允许跨域
    res.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
    return res




class SecondThread(threading.Thread):
    def __init__(self):
        super(SecondThread, self).__init__()  # 注意：一定要显式的调用父类的初始化函数。
        # self.arg=arg

    def run(self):  # 定义每个线程要运行的函数
        print('second thread is run!')
        global shared_image
        while True:
            camera = cv2.VideoCapture(RTMP_HOST)
            if (camera.isOpened()):
                print ('Open camera 0')
                break
            else:
                print ('Fail to open camera 0!')
                time.sleep(0.05)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # 2560x1920 2217x2217 2952×1944 1920x1080
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        # camera.set(cv2.CAP_PROP_FPS, 5)

        while True:
            ret, frame = camera.read()  # 逐帧采集视频流

            if frame is not None:
                image = Image.fromarray(frame)
                image = image.resize((960, 576))
                frame = np.array(image)

                # frame.resize((960, 540))
                if use_channel == 1:
                    shared_image = frame





class TFThread(threading.Thread):
    def __init__(self):
        super(TFThread, self).__init__()  # 注意：一定要显式的调用父类的初始化函数。
        # self.arg=arg

    def run(self):  # 定义每个线程要运行的函数
        print('tensorflow thread is run!')
        global shared_image
        global process_image
        global people_count
        while True:
            frame, pc = od.infer(shared_image)
            pipe.stdin.write(frame.tobytes())
            people_count = pc
            # print(process_image)


command = ['ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec','rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', '960x576',
    '-r', str(5),
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'flv',
    rtmpUrl]

global pipe
pipe = sp.Popen(command, stdin=sp.PIPE)


class PushThread(threading.Thread):
    def __init__(self):
        super(PushThread, self).__init__()  # 注意：一定要显式的调用父类的初始化函数。
        # self.arg=arg

    def run(self):  # 定义每个线程要运行的函数
        print('push thread is run!')
        global process_image
        url = "http://127.0.0.1:8080/PeopleDetection/people"
        count = 0
        while True:
            ###########################图片采集
            #print(process_image)
            #print(pipe)
            #print(pipe.stdin)
            # pipe.stdin.write(process_image.tostring())  # 存入管道
            # print('push!')
            param = {'peopleNum': str(people_count)}
            count += 1
            if count % 25 == 0:
                try:
                    r = requests.post(url=url, data=param)
                except:
                    pass
            time.sleep(0.198)


class GetChannelThread(threading.Thread):
    def __init__(self):
        super(GetChannelThread, self).__init__()  # 注意：一定要显式的调用父类的初始化函数。
        # self.arg=arg

    def run(self):  # 定义每个线程要运行的函数
        print('get channel thread is run!')
        global use_channel
        url = 'http://127.0.0.1:8080/PeopleDetection/get_channel'
        while True:
            try:
                r = requests.get(url=url)
                use_channel = int(eval(r.content)['data'])
                print('当前通道：' + str(use_channel))
            except:
                pass
            time.sleep(5)



second_thread = SecondThread()
second_thread.start()

tf_thread = TFThread()
tf_thread.start()

push_thread = PushThread()
push_thread.start()

get_channel_thread = GetChannelThread()
app.run(host='0.0.0.0', port=5000, debug = True)
#get_channel_thread.start()

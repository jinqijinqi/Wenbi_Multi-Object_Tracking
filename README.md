Author: Wenbin Luo,

Email: 349057878@qq.com

Current Affiliation: UESTC

Future Affiliation:  Tencent Company

supervisor: Jin Qi

Date: 5/30/2022

note: this work was done while I pursued my master degree in Jin Qi's AIML Lab in UESTC

# streamlit-model-serving

Simple example of usage of streamlit for My model serving.

To run the example in a machine running Docker and docker-compose, run:

    docker-compose build
    docker-compose up

To visit the streamlit UI, visit http://localhost:8501.

Logs can be inspected via:

    docker-compose logs

### Deployment

To deploy the app, one option is deployment on Heroku (with [Dockhero](https://elements.heroku.com/addons/dockhero)). To do so:

- rename `docker-compose.yml` to `dockhero-compose.yml`
- create an app (we refer to its name as `<my-app>`) on a Heroku account
- install locally the Heroku CLI, and enable the Dockhero plugin with `heroku plugins:install dockhero`
- add to the app the DockHero add-on (and with a plan allowing enough RAM to run the model!)
- in a command line enter `heroku dh:compose up -d --app <my-app>` to deploy the app
- to find the address of the app on the web, enter `heroku dh:open --app <my-app>`
- visit the address adding `:8501` to visit the streamlit interface
- logs are accessible via `heroku logs -p dockhero --app <my-app>`

如果DCNv2运行时报错： RuntimeError: Not compiled with GPU support
进入docker terminal ：
```
    cd DCNv2-pytorch_1.7
    ./make.sh
```
重新编译即可


###Train&&Test

#### 环境：

1.安装conda 

```
bash Miniconda3-latest-Linux-x86_64.run
cuda_10.2.89_440.33.01_linux.run(只安装tool)

```

重启shell

2.创建虚拟环境

```
conda create -n mot
conda activate mot
```

3.安装环境依赖

```
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd ${MOT_ROOT}
pip install cython
pip install mvcc
pip install -r requirements.txt
pip install timm

git clone -b pytorch_1.7 https://github.com/ifzhang/DCNv2.git
cd DCNv2
./make.sh

```

如果dcnv2安装不顺利可以使用mmcv库中的DCN模块代替DCNv2官方库，使用起来非常简单，如下：

1. 安装mmcv库：

```bash
# 命令行输入：
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# 将其中的{cu_version}替换为你的CUDA版本，{torch_version}替换为你已经安装的pytorch版本；
# 例如：CUDA 为11.0，pytorch为1.7.0
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

\2. 在代码中使用DCN：

```python
from mmcv.ops import DeformConv2dPack as DCN

# 使用方法与官方DCNv2一样，只不过deformable_groups参数名改为deform_groups即可，例如：
dconv2 = DCN(in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=1, deform_groups=2)
```

4.为了运行演示代码，您还需要安装[ffmpeg](https://www.ffmpeg.org/)。

缺少依赖包时sudo apt-get build-dep ffmpeg

#### 数据集准备：

- **CrowdHuman** CrowdHuman 数据集可以从他们的[官方网页](https://www.crowdhuman.org/)下载。下载后，您应该按照以下结构准备数据：

```
crowdhuman
   |——————images
   |        └——————train
   |        └——————val
   └——————labels_with_ids
   |         └——————train(empty)
   |         └——————val(empty)
   └------annotation_train.odgt
   └------annotation_val.odgt
```

如果您想在 CrowdHuman 上进行预训练（我们在 CrowdHuman 上训练 Re-ID），您可以更改 src/gen_labels_crowd_id.py 中的路径并运行：

```
cd src
python gen_labels_crowd_id.py
```

如果您想将 CrowdHuman 添加到 MIX 数据集（我们不在 CrowdHuman 上训练 Re-ID），您可以更改 src/gen_labels_crowd_det.py 中的路径并运行：

```
cd src
python gen_labels_crowd_det.py
```

- **MIX** 我们在这部分使用与[JDE](https://github.com/Zhongdao/Towards-Realtime-MOT)相同的训练数据，我们称之为“MIX”。请参考他们的[DATA ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md)下载并准备所有训练数据，包括 Caltech Pedestrian、CityPersons、CUHK-SYSU、PRW、ETHZ、MOT17 和 MOT16。
- **2DMOT15和MOT20** [2DMOT15](https://motchallenge.net/data/2D_MOT_2015/)和[MOT20](https://motchallenge.net/data/MOT20/)可以在MOT挑战官网下载。下载后，您应该按照以下结构准备数据：

```
MOT15
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT20
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
```

然后，您可以更改 src/gen_labels_15.py 和 src/gen_labels_20.py 中的 seq_root 和 label_root 并运行：

```
cd src
python gen_labels_15.py
python gen_labels_20.py
```

生成 2DMOT15 和 MOT20 的标签。2DMOT15的seqinfo.ini文件可以在[[Google\]](https://drive.google.com/open?id=1kJYySZy7wyETH4fKMzgJrYUrTfxKlN1w)、[[Baidu\]下载，代码：8o0w](https://pan.baidu.com/s/1zb5tBW7-YTzWOXpd9IzS0g)。


#### Training
* Download the training data
* Change the dataset root directory 'root' in src/lib/cfg/data.json and 'data_dir' in src/lib/opts.py
* Train det on  Bytest(CrowdHuman+MIX) and Train reid on MOT17:
```
sh experiments/mot17_mit_1.sh
```

#### Tracking
* For ablation study, we evaluate on the other half of the training set of MOT17, you can run:
```
cd src
python track_half.py mot --load_model ../exp/models/best.pth --arch mit_1 --conf_thres 0.4 --val_mot17 True
```
* To get the txt results of the test set of MOT16 or MOT17, you can run:
```
cd src
python track.py mot --test_mot17 True --load_model ../exp/models/best.pth --arch mit_1 --conf_thres 0.4
python track.py mot --test_mot16 True --load_model ../exp/models/best.pth --arch mit_1 --conf_thres 0.4
```

and send the txt files to the [MOT challenge](https://motchallenge.net) evaluation server to get the results. (You can get the SOTA results 74+ MOTA on MOT17 test set using the baseline model 'best.pth'.)

#### Demo
You can input a raw video and get the demo video by running src/demo.py and get the mp4 format of the demo video:
```
cd src
python demo.py mot --load_model ../exp/models/best.pth --arch mit_1 --conf_thres 0.4
```
You can change --input-video and --output-root to get the demos of your own videos.
--conf_thres can be set from 0.3 to 0.7 depending on your own videos.

#### Train on custom dataset
You can train FairMOT on custom dataset by following several steps bellow:
1. Generate one txt label file for one image. Each line of the txt label file represents one object. The format of the line is: "class id x_center/img_width y_center/img_height w/img_width h/img_height". You can modify src/gen_labels_16.py to generate label files for your custom dataset.
2. Generate files containing image paths. The example files are in src/data/. Some similar code can be found in src/gen_labels_crowd.py
3. Create a json file for your custom dataset in src/lib/cfg/. You need to specify the "root" and "train" keys in the json file. You can find some examples in src/lib/cfg/.
4. Add --data_cfg '../src/lib/cfg/your_dataset.json' when training. 




#### Evaluation Measures

Lower is better. Higher is better.

| **Measure** | **Better** | **Perfect** | **Description**                                              |
| ----------- | ---------- | ----------- | ------------------------------------------------------------ |
| **MOTA**    | higher     | 100%        | Multi-Object Tracking Accuracy (+/- denotes standard deviation across all sequences) [1]. This measure combines three error sources: false positives, missed targets and identity switches. |
| **IDF1**    | higher     | 100%        | ID F1 Score [2]. The ratio of correctly identified detections over the average number of ground-truth and computed detections. |
| **HOTA**    | higher     | 100%        | Higher Order Tracking Accuracy [3]. Geometric mean of detection accuracy and association accuracy. Averaged across localization thresholds. |
| **MT**      | higher     | 100%        | Mostly tracked targets. The ratio of ground-truth trajectories that are covered by a track hypothesis for at least 80% of their respective life span. |
| **ML**      | lower      | 0%          | Mostly lost targets. The ratio of ground-truth trajectories that are covered by a track hypothesis for at most 20% of their respective life span. |
| **FP**      | lower      | 0           | The total number of false positives.                         |
| **FN**      | lower      | 0           | The total number of false negatives (missed targets).        |
| **Rcll**    | higher     | 100%        | Ratio of correct detections to total number of GT boxes.     |
| **Prcn**    | higher     | 100%        | Ratio of TP / (TP+FP).                                       |
| **AssA**    | higher     | 100%        | Association Accuracy [3]. Association Jaccard index averaged over all matching detections and then averaged over localization thresholds. |
| **DetA**    | higher     | 100%        | Detection Accuracy [3]. Detection Jaccard index averaged over localization thresholds. |
| **AssRe**   | higher     | 100%        | Association Recall [3]. TPA / (TPA + FNA) averaged over all matching detections and then averaged over localization thresholds. |
| **AssPr**   | higher     | 100%        | Association Precision [3]. TPA / (TPA + FPA) averaged over all matching detections and then averaged over localization thresholds. |
| **DetRe**   | higher     | 100%        | Detection Recall [3]. TP /(TP + FN) averaged over localization thresholds. |
| **DetPr**   | higher     | 100%        | Detection Precision [3]. TP /(TP + FP) averaged over localization thresholds. |
| **LocA**    | higher     | 100%        | Localization Accuracy [3]. Average localization similarity averaged over all matching detections and averaged over localization thresholds. |
| **FAF**     | lower      | 0           | The average number of false alarms per frame.                |
| **ID Sw.**  | lower      | 0           | Number of Identity Switches (ID switch ratio = #ID switches / recall) [4]. Please note that we follow the stricter definition of identity switches as described in the reference |
| **Frag**    | lower      | 0           | The total number of times a trajectory is fragmented (i.e. interrupted during tracking). |
| **Hz**      | higher     | Inf.        | Processing speed (in frames per second excluding the detector) on the benchmark. The frequency is provided by the authors and not officially evaluated by the MOTChallenge. |


### Debugging

To modify and debug the app, [development in containers](https://davidefiocco.github.io/debugging-containers-with-vs-code) can be useful (and kind of fun!).

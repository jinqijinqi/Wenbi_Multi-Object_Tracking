cd src
python train.py mot --exp_id cp_data_dla34 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/data_cp.json' --fp16 True   
#python train.py mot --exp_id reid_dla34 --load_model '/root/data/lwb/FairMOT/exp/mot/cp_data_dla34/model_30.pth' --data_cfg '../src/lib/cfg/mot17_half.json' --fp16 True  --only_reid True --num_epochs 10 --lr_step '5'
cd ..
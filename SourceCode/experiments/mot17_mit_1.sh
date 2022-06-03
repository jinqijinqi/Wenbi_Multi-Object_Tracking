cd src
python train.py mot --exp_id all_mit_1 --data_cfg '../src/lib/cfg/bytest.json' --arch mit_1  --fp16 True  --batch_size 8 --only_dect True --num_epochs 60 --lr_step '50'
python train.py mot --exp_id reid_mit_all_mot17  --load_model '/root/data/lwb/FairMOT/exp/mot/all_mit_1/model_60.pth' --data_cfg '../src/lib/cfg/mot17.json' --arch mit_1  --batch_size 12  --fp16 True  --only_reid True --num_epochs 20 --lr_step '15'
cd ..
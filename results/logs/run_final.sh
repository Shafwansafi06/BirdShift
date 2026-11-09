cd ~/Desktop/Shafwan_CV_project/BirdShift
export PYTHONPATH=src
.venv/bin/python scripts/train.py --mode full --epochs 50 --image-size 448 --amp --workers 8 --train-aug fg --scheduler cosine --optimizer sgd --lr 0.01 --head-lr-mult 10 --weights resnet50.tv_in1k --train-split trainval --run-name final_sgd_tv > results/logs/final_sgd_tv.log 2>&1
cp checkpoints/final_sgd_tv/last.pth checkpoints/final_sgd_tv/best.pth
.venv/bin/python scripts/eval_test.py final_sgd_tv > results/logs/final_test.log 2>&1
echo done > results/logs/run_final.status

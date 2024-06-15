device=$1
python -u train.py --dataset acm --source acm --target dblp --epoch 500 --alpha 30 --method mmd-un --gpu_id $device
python -u train.py --dataset acm --source dblp --target acm --epoch 500 --alpha 30 --method mmd-un --gpu_id $device
python -u train.py --dataset acm --source acm --target citation --epoch 500 --method mmd-un --gpu_id $device
python -u train.py --dataset acm --source citation --target acm --epoch 500 --alpha 30 --method mmd-un --gpu_id $device
python -u train.py --dataset acm --source dblp --target citation --epoch 500 --alpha 30 --method mmd-un --gpu_id $device
python -u train.py --dataset acm --source citation --target dblp --epoch 500 --alpha 30 --method mmd-un --gpu_id $device
# cd logs/detection_dataset
for m in $(ls logs/trash/bar)
do
#   cd ${c}
#   pwd
  echo $C
  python3 coco_eval.py -w logs/trash/bar/${m}
#   cd ..
done
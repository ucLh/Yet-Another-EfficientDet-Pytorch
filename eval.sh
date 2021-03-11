python coco_eval.py \
--weights weights/efficientdet-d0_original_pretrain.pth \
--use_coco_classes

python coco_eval.py \
--weights weights/efficientdet-d0_noise_pretrain.pth \
--use_coco_classes

python coco_eval.py \
--weights weights/efficientdet-d0_fixed_pretrain.pth \
--use_coco_classes

python coco_eval.py \
--weights weights/efficientdet-d0_original_from_zero.pth

python coco_eval.py \
--weights weights/efficientdet-d0_noise_from_zero.pth

python coco_eval.py \
--weights weights/efficientdet-d0_fixed_from_zero.pth
PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env main.py \
--pretrained params/detr-r50-pre-2stage-q64.pth \
--output_dir logs_new \
--dataset_file hico \
--hoi_path data/hico_20160224_det \
--num_obj_classes 80 \
--num_verb_classes 117 \
--num_hoi_classes 600 \
--backbone resnet50 \
--num_queries 64 \
--dec_layers_hopd 3 \
--dec_layers_interaction 3 \
--epochs 90 \
--lr_drop 60 \
--use_nms_filter

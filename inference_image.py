from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
# config_file = 'configs/yolact/yolact_r101_1x8_coco.py'
checkpoint_file = 'latest.pth'
# checkpoint_file = 'work_dirs/yolact_r101_1x8_coco/latest.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'bunchofcows2.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='mask2former_result.jpg')
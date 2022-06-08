from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
checkpoint_file = 'iter_185000.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a video and show the results
video = mmcv.VideoReader('data/test_videos/IMG_4172-hevcmp4.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
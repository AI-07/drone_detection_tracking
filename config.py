
MAX_ALLOWED_RES = 1200
CROP_CENTER = False
# Center cropping dimensions
CROP_CENTER_WIDTH = 400
CROP_CENTER_HEIGHT = 400

FOV = 180                                       # Camera's Field-of-View
TRACKER_TYPE = "MOSSE"                          # Choose from [ "MIL", "BOOSTING", "KCF", "TLD", "MEDIANFLOW", "MOSSE", "CSRT" ]
VIDEO_PATH = "../input_videos/vid_out.mp4"


# Path to the YOLO model
MODEL_PATH = 'model_files/drone_detection_60E_n_saved_model/drone_detection_60E_n_full_integer_quant.tflite'
VISUALIZE_FRAME = True
SAVE_OUTPUT_VIDEO = False



INFO_COLOR = (255, 255, 255)
INFO_RECT_COLOR = (0, 0, 0)
TRAINING_RES = 640
DETECTION_INTERVAL = 30                         # No of frames after which detection occurs


OUTPUT_PATH_TEMPLATE = "runs/{tracker_type}_{video_name}.mp4"




RESIZE_FLAG = False


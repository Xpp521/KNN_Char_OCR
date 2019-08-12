from os.path import join

# Data folder.
DATA_FOLDER = 'data'

# Image folder.
IMG_FOLDER = 'img'

# Sample data path.
DATA_PATH = join(DATA_FOLDER, 'knn_data.npz')

# Initial sample data path.
INIT_DATA_PATH = join(DATA_FOLDER, 'digits_reverse.png')

# Whether to turn on video.
RECORD = 0

# Number of used nearest neighbors. Should be greater than 1.
K = 3   # 5

# Camera URI.
# e.x.: 'http://username:password@ip:port/video'
# if you want to use system default camera,set it to 0.
CAMERA_URI = 0
# CAMERA_URI = 'http://username:password@192.168.1.100:6666/video'

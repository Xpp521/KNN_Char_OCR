import cv2
import numpy as np

from os import mkdir
from shutil import rmtree
from os.path import exists, join

from utils import open_file
from setting import CAMERA_URI, RECORD, IMG_FOLDER, DATA_PATH, INIT_DATA_PATH, K


def load_knn_data(filepath):
    """
    Load kNN sample data from file.
    :param filepath: the file path.
    :return:
    """
    with np.load(filepath) as data:
        print('Loading data： {}'.format(data.files))
        train = data.get('train')
        train_label = data.get('train_label')
    return update_knn(cv2.ml.KNearest_create(), train, train_label)


def save_knn_data(filepath, train, train_label):
    """
    Save KNN sample data.
    :param filepath: the file path.
    :param train: sample data.
    :param train_label: sample data label.
    """
    np.savez(filepath, train=train, train_label=train_label)


def create_knn(init_data_path):
    """
    初始化KNN算法。
    :return:
    """
    knn = cv2.ml.KNearest_create()
    img = cv2.imread(init_data_path)                # 读取初始数据
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 转成灰度图
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]    # 划分图片
    train = np.array(cells).reshape(-1, 400).astype(np.float32)
    train_label = np.repeat(np.arange(10), 500)
    return update_knn(knn, train, train_label)


def update_knn(knn, train, train_label, new_data=None, new_data_label=None):
    """
    Add new sample data.
    :param knn: KNN algorithm.
    :param train: train data.
    :param train_label: train label.
    :param new_data:
    :param new_data_label:
    :return:
    """
    if new_data is not None and new_data_label is not None:
        print(train.shape, new_data.shape)
        new_data = new_data.reshape(-1, 400).astype(np.float32)
        train = np.vstack((train, new_data))
        train_label = np.hstack((train_label, new_data_label))
    knn.train(train, cv2.ml.ROW_SAMPLE, train_label)
    return knn, train, train_label


def find_roi(frame, thres_value):
    rois = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
    gray2 = cv2.dilate(gray, None, iterations=2)    # 膨胀2次
    gray2 = cv2.erode(gray2, None, iterations=2)    # 腐蚀2次
    edges = cv2.absdiff(gray, gray2)                # 做差
    # 用Sobel算子边缘检测
    x = cv2.Sobel(edges, cv2.CV_16S, 1, 0)          # 计算x方向梯度
    y = cv2.Sobel(edges, cv2.CV_16S, 0, 1)          # 计算y方向梯度
    abs_x = cv2.convertScaleAbs(x)                  # x方向梯度图像取绝对值
    abs_y = cv2.convertScaleAbs(y)                  # y方向梯度图像取绝对值
    dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)    # 融合x、y两个梯度图像
    ret, ddst = cv2.threshold(edges, thres_value, 255, cv2.THRESH_BINARY)                     # 二值化
    contours, hierarchy = cv2.findContours(ddst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找边界
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 20:
            rois.append((x, y, w, h))
    return rois, ddst


def find_digit(knn, roi, thres_value):
    # ret, th = cv2.threshold(roi, thres_value, 255, cv2.THRESH_BINARY)   # 转换成二进制图像
    th = cv2.resize(roi, (20, 20))                                       # 调整尺寸
    out = th.reshape(-1, 400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(out, k=K)
    return int(result[0][0]), th


def concatenate(images, direction='h'):
    """
    Stitching multiple images.
    :param images: a list of multiple image arrays.
    :param direction: stitching direction, value: 'v' present vertical,'h' present horizontal.
    :return: result image array.
    """
    if 'h' == direction:
        output = np.zeros(20 * 20 * len(images), np.float32).reshape(20, -1)
        for index, img in enumerate(images):
            output[:, 20 * index:20 * (index + 1)] = img
    else:
        output = np.zeros(20 * 20 * len(images),  np.float32).reshape(-1, 20)
        for index, img in enumerate(images):
            output[20 * index:20 * (index + 1), :] = img
    return output


def main():
    mkdir(IMG_FOLDER)
    if exists(DATA_PATH):
        knn, train, train_label = load_knn_data(DATA_PATH)
    elif exists(INIT_DATA_PATH):
        knn, train, train_label = create_knn(INIT_DATA_PATH)
    else:
        return
    print(knn.getAlgorithmType())
    print(knn.getDefaultK())
    print('Connecting camera: {} ...'.format(CAMERA_URI))
    cap = cv2.VideoCapture(CAMERA_URI)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = 426
    # height = 480
    video_writer = cv2.VideoWriter('frame.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                                   (int(width) * 2, int(height)), True) if RECORD else None
    count = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('Camera status: disconnected!')
            break
        frame = frame[:, :width]
        rois, dst = find_roi(frame, 50)
        digits = []
        for (x, y, w, h) in rois:
            digit, th = find_digit(knn, dst[y:y + h, x:x + w], 50)
            digits.append(cv2.resize(th, (20, 20)))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (153, 153, 0), 2)
            cv2.putText(frame, str(digit), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 0, 255), 2)
        new_edges = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        new_frame = np.hstack((frame, new_edges))
        cv2.imshow('frame', new_frame)
        if video_writer:
            video_writer.write(new_frame)
        key = cv2.waitKey(1) & 0xff
        if key == 27:                        # the keyValue of "esc" is 27
            break
        elif key == ord(' '):
            nd = len(digits)
            output = concatenate(digits)
            show_digits = cv2.resize(output, (60 * nd, 60))
            # cv2.imshow('digits', show_digits.astype(np.uint8))
            # cv2.waitKeyEx()
            filepath = join(IMG_FOLDER, '{}.png'.format(count))
            cv2.imwrite(filepath, show_digits)
            count += 1
            open_file(filepath)                 # show image.
            numbers = list(input('input the digits:'))
            if nd != len(numbers):
                print('update KNN fail!')
                continue
            try:
                knn, train, train_label = update_knn(knn, train, train_label, output, [int(i) for i in numbers])
                print('update KNN, Done!')
            except ValueError:
                print('update KNN fail!')
                continue
            # finally:
                # cv2.destroyWindow('digits')
    print('Numbers of trained images:', len(train))
    print('Numbers of trained image labels', len(train_label))
    cap.release()
    cv2.destroyAllWindows()

    save_knn_data(DATA_PATH, train, train_label)
    rmtree(IMG_FOLDER)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        rmtree(IMG_FOLDER)

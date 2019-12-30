# python 3.7.4
# opencv-contrib 4.1.2
# opencv 4.1.1
# CUDA 10.0
# dlib 19.18.0
# face_recognition 1.2.3


import pathlib
import cv2
import face_recognition as fr
import numpy as np
import tensorflow as tf

# 人脸识别模型选择，hog为Histogram of Oriented Gradients梯度方向直方图模型，cnn为Convolutional Neural Networks卷积神经网络模型
# hog速度快，但精准度较低，cnn即使用了GPU加速仍然速度慢，但精准度高
FACE_LOCATION_MODEL = 'hog'
# 人脸编码时上采样次数，越大越精准，但越慢
IMAGE_JITTERS = 5
# 视频采样间隔帧数，每隔几帧进行一次人脸检测和识别
FRAME_DETECT_RATE = 2
# 视频下采样比例，将视频中的图片长宽各除以FRAME_RESIZE_SCALE，加快模型速度
FRAME_RESIZE_SCALE = 2
# 人脸识别特征比较阈值，越小越容易判断为同一人
FACE_DISTANCE_TOLERANCE = 0.6
# 存放人脸图像的文件夹路径
KNOWN_IMAGE_PATH = 'datasets/faces'
# 加载情绪识别模型
emotion_classifier = tf.keras.models.load_model('emotion.hdf5')
emotion_labels = {
    0: 'angry',
    1: 'hate',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'calm'
}


def test():
    # 打开摄像头
    video_capture = cv2.VideoCapture(0)

    # 用pathlib获取目标文件夹下所有图片，并使用文件名来作为人名
    # opencv的putText函数只支持英文绘制，中文会出现乱码，有空可以换用Pillow绘制中文名
    data_root = pathlib.Path(KNOWN_IMAGE_PATH)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_images = [fr.load_image_file(path) for path in all_image_paths]

    known_face_names = [pathlib.Path(path).stem for path in all_image_paths]
    # 找到图片中的人脸区域，并对人脸区域进行采样编码，每张人脸共128个特征编码
    known_face_locations = [fr.face_locations(image, 1, model=FACE_LOCATION_MODEL) for image in all_images]
    known_face_encodings = [fr.face_encodings(image, location, num_jitters=IMAGE_JITTERS)[0]
                            for image, location in zip(all_images, known_face_locations)]

    print(len(known_face_encodings))
    print(len(known_face_names))

    # 初始化帧数计数器
    frame_count = 0
    while True:
        # 从摄像头中获取一帧，ret为视频文件末尾标识符，在摄像头实时采集情况下无用，frame即为一帧，为3维数组
        ret, frame = video_capture.read()

        # 当计数器到采样帧数时，进行采样及人脸识别标注
        if frame_count % FRAME_DETECT_RATE == 0:
            frame_count = 0

            # 将视频帧长度与宽度分别变为FRAME_RESIZE_SCALE分之一的大小
            small_frame = cv2.resize(frame, (0, 0), fx=1 / FRAME_RESIZE_SCALE, fy=1 / FRAME_RESIZE_SCALE)

            # 将视频帧的RGB色彩从opencv表示方法转为dlib模型表示方法
            rgb_small_frame = small_frame[:, :, ::-1]

            # 找到一帧图像中所有人脸，并分别进行编码
            face_locations = fr.face_locations(rgb_small_frame, model=FACE_LOCATION_MODEL)
            face_encodings = fr.face_encodings(rgb_small_frame, face_locations, num_jitters=IMAGE_JITTERS)

            face_names = []
            face_emotions = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # 默认人名为unknown
                name = 'unknown'

                # 计算识别出的人脸与已知人脸的编码距离，将其中距离最小，且小于阈值的人脸来作为识别结果
                face_distances = fr.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= FACE_DISTANCE_TOLERANCE:
                    name = known_face_names[best_match_index]
                face_names.append(name)

                # 检测表情
                face_emotion = frame[top:bottom, left:right]
                face_emotion = cv2.cvtColor(face_emotion, cv2.COLOR_BGR2GRAY)
                face_emotion = cv2.resize(face_emotion, (48, 48))
                face_emotion = face_emotion / 255.0
                face_emotion = np.expand_dims(face_emotion, 0)
                face_emotion = np.expand_dims(face_emotion, -1)
                emotion_label_arg = np.argmax(emotion_classifier.predict(face_emotion))
                emotion = emotion_labels[emotion_label_arg]
                face_emotions.append(emotion)

        # 标识人脸识别结果
        for (top, right, bottom, left), name, emotion in zip(face_locations, face_names, face_emotions):
            # 将识别出的人脸位置坐标缩放回原先的比例
            top *= FRAME_RESIZE_SCALE
            right *= FRAME_RESIZE_SCALE
            bottom *= FRAME_RESIZE_SCALE
            left *= FRAME_RESIZE_SCALE

            # 使用一个矩形标识识别出的人脸区域
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

            # 在矩形框下方绘制识别出的人名
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, name, (left + 6, bottom + 20), font, 1.0, (255, 255, 255), 1)
            cv2.putText(frame, emotion, (left + 6, bottom + 40), font, 1.0, (255, 255, 255), 1)

        # 显示识别结果
        cv2.imshow('Video', frame)

        # 用q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 释放摄像头并关闭所有图像窗口
    video_capture.release()
    cv2.destroyAllWindows()
    pass


if __name__ == '__main__':
    test()

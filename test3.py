import cv2
import numpy as np
import pyrealsense2 as rs

'''
设置
'''
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流

# config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)  # streaming流开始

# 创建对齐对象与color流对齐
align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐

''' 
获取对齐图像帧与相机参数
'''


def get_aligned_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    #### 将images转为numpy arrays ####
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame


'''
canny边缘检测
'''


def canny_detection(img):
    canny = cv2.Canny(img, 100, 200)
    res = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)
    return res


'''
sobel边缘检测
'''


def sobel_detection(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


'''
laplacian边缘检测
'''


def laplacian_detection(img):
    out = cv2.GaussianBlur(img, (3, 3), 1.3)
    gray_lap = cv2.Laplacian(out, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(gray_lap)
    return dst


'''
harris角点检测
'''


def harris_detection(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(imgray)
    dst = cv2.cornerHarris(gray, 8, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.005 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # 图像连通域
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # 迭代停止规则
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    for i in res:
        x1, y1, x2, y2 = i.ravel()
        cv2.circle(img, (x1, y1), 3, 255, -1)
        cv2.circle(img, (x2, y2), 3, (0, 255, 0), -1)
    image = img[:, :, ::-1]
    image = image.copy()
    return image


'''
fast角点检测
'''


def fast_detection(img):
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img, None)

    frame = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
    image = frame[:, :, ::-1]
    image = image.copy()
    return image


'''
tomasi角点检测
'''


def tomasi_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 72, 0.01, 10)  # 棋盘上的所有点

    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)  # 在原图像上画出角点位置
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    image = img[:, :, ::-1]
    image = image.copy()
    return image


'''
hough圆检测
'''


def hough_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find circles with HoughCircles
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, minDist=150, circles=None, param1=200, param2=18,
                               maxRadius=40, minRadius=20)

    # Draw circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # if r > 30:
            cv2.circle(img, (x, y), r, (36, 255, 12), 3)
    image = img[:, :, ::-1]
    image = image.copy()
    return image


if __name__ == '__main__':
    while True:
        ''' 
        获取对齐图像帧与相机参数
        '''
        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        canny_edge = canny_detection(img_color)
        cv2.putText(canny_edge, "Canny Edge Detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [0, 0, 255])
        sobel_edge = sobel_detection(img_color)
        cv2.putText(sobel_edge, "Sobel Edge Detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [0, 0, 255])
        laplacian_edge = laplacian_detection(img_color)
        cv2.putText(laplacian_edge, "Laplacian Edge Detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [0, 0, 255])

        edge = np.hstack([canny_edge, sobel_edge, laplacian_edge])

        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        harris_corner = harris_detection(img_color)
        cv2.putText(harris_corner, "Harris Corner Detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [0, 0, 255])
        fast_corner = fast_detection(img_color)
        cv2.putText(fast_corner, "FAST Corner Detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [0, 0, 255])
        tomasi_corner = tomasi_detection(img_color)
        cv2.putText(tomasi_corner, "Tomasi Corner Detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [0, 0, 255])
        corner = np.hstack([harris_corner, fast_corner, tomasi_corner])

        color_intrin, depth_intrin, img_color, img_depth, aligned_depth_frame = get_aligned_images()  # 获取对齐图像与相机参数
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        hough_circle = hough_detection(img_color)
        cv2.putText(hough_circle, "Hough Circle Detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [0, 0, 255])

        circle = np.hstack([hough_circle])

        # res = np.vstack([edge, corner])

        #### 显示画面 ####
        # # cv2.imshow('edge_detection', edge)
        cv2.imshow('corner_detection', corner)
        # cv2.imshow('circle_detection', circle)
        key = cv2.waitKey(100)

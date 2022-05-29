# -*- coding: utf-8 -*-
import math

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from skimage import exposure

''' 
设置
'''
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  # 配置depth流
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 配置color流

# config.enable_stream(rs.stream.depth,  848, 480, rs.format.z16, 90)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipe_profile = pipeline.start(config)  # streaming流开始

# # 创建对齐对象与color流对齐
# align_to = rs.stream.color  # align_to 是计划对齐深度帧的流类型
# align = rs.align(align_to)  # rs.align 执行深度帧与其他帧的对齐

''' 
获取对齐图像帧与相机参数
'''


def get_images():
    frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集

    depth_frame = frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    color_frame = frames.get_color_frame()  # 获取对齐帧中的的color帧

    #### 获取相机参数 ####
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    #### 将images转为numpy arrays ####
    img_color = np.asanyarray(color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(depth_frame.get_data())  # 深度图（默认16位）

    return color_intrin, depth_intrin, img_color, img_depth, depth_frame


''' 
获取随机点三维坐标
'''


def get_3d_camera_coordinate(depth_pixel, depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate


'''
两点距离
'''


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2) + math.pow((p2[2] - p1[2]), 2))


'''
内径测量
'''


def hough_detection(img, depth_frame, depth_intrin):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find circles with HoughCircles
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, minDist=100, param1=50, param2=30, minRadius=20,
                               maxRadius=40)

    # Draw circles
    radius = math.inf
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for circle in circles:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            cv2.circle(img, (x, y), r, (36, 255, 12), 3)
            if (x - r) >= 1280 or (x + r) >= 1280 or y >= 720 or (x - r) <= 0 or y <= 0:
                continue
            p1 = [x - r, y]
            p2 = [x + r, y]
            dis1, c1 = get_3d_camera_coordinate(p1, depth_frame, depth_intrin)
            dis2, c2 = get_3d_camera_coordinate(p2, depth_frame, depth_intrin)
            dist = cal_distance(c1, c2)
            if dist < radius:
                radius = dist
    return radius


'''
矩阵中行向量两两距离
'''


def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


'''
外径、长、宽测量
'''


def corner_detection(img, depth_frame, depth_intrin):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # harris角点检测图像需为float32
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

    corner_axis = []

    for i in res:
        x1, y1, x2, y2 = i.ravel()
        point = [x1, y1]
        dis, coordinate_3d = get_3d_camera_coordinate(point, depth_frame, depth_intrin)
        corner_axis.append(coordinate_3d)

    corners = np.array(corner_axis)
    d = EuclideanDistances(corners, corners)
    d = torch.from_numpy(d)
    r1 = torch.argmax(d)
    a = r1.item() / 2
    b = 2 * a * math.cos(math.radians(30))
    return r1.item(), a, b


if __name__ == "__main__":

    while True:
        ''' 
        获取对齐图像帧与相机参数
        '''
        color_intrin, depth_intrin, img_color, img_depth, depth_frame = get_images()  # 获取对齐图像与相机参数

        r1, a, b = corner_detection(img_color, depth_frame, depth_intrin)
        r2 = hough_detection(img_color, depth_frame, depth_intrin)

        ''' 
        获取随机点三维坐标
        '''
        depth_pixel = [640, 360]  # 设置随机点，以相机中心点为例
        dis, camera_coordinate = get_3d_camera_coordinate(depth_pixel, depth_frame, depth_intrin)

        ''' 
        显示图像与标注
        '''
        #### 在图中标记随机点及其坐标 ####
        cv2.circle(img_color, (640, 360), 8, [255, 0, 255], thickness=-1)

        print("+++++++++++")
        print(dis)
        print(r1)
        print(r2)
        print(a)
        print(b)
        print("+++++++++++")

        cv2.putText(img_color, "d = " + str(format(dis, ".2f")) + " cm", (80, 80), cv2.FONT_HERSHEY_PLAIN,
                    2,
                    [0, 0, 255])
        cv2.putText(img_color, "r1 = " + str(format(r1, ".2f")) + " mm", (80, 120), cv2.FONT_HERSHEY_PLAIN, 2,
                    [0, 0, 255])
        cv2.putText(img_color, "r2 = " + str(format(r2, ".2f")) + " mm", (80, 160), cv2.FONT_HERSHEY_PLAIN, 2,
                    [0, 0, 255])
        cv2.putText(img_color, "a = " + str(format(a, ".2f")) + " mm", (80, 200), cv2.FONT_HERSHEY_PLAIN, 2,
                    [0, 0, 255])
        cv2.putText(img_color, "b = " + str(format(b, ".2f")) + " mm", (80, 240), cv2.FONT_HERSHEY_PLAIN, 2,
                    [0, 0, 255])

        # gam1 = exposure.adjust_gamma(img_color, 0.5)  # 调暗
        # cv2.resize(gam1, (120, 120))
        # gam2 = exposure.adjust_gamma(img_color, 5)
        # cv2.resize(gam2, (120, 120))
        cv2.resize(img_color, (30, 30))
        # res = np.hstack([gam1, img_color, gam2])

        #### 显示画面 ###
        cv2.namedWindow("Measurement", 0)
        cv2.imshow('Measurement', img_color)
        key = cv2.waitKey(1)

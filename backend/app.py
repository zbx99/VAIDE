import base64
import math
import uuid

import shortuuid
from flask import Flask,request,jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.morphology import skeletonize
from operator import itemgetter

app = Flask(__name__)
# 允许跨域传输数据
CORS(app)


def get_color_mask(img,lower,upper,debug):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 根据颜色阈值提取目标图像
    mask = cv2.inRange(hsv, lower, upper)

    # 将目标图像与原图像进行与运算，提取出颜色符合条件的图像
    result = cv2.bitwise_and(img, img, mask=mask)
    if debug:
        cv2.imshow("result_green", result)
        cv2.waitKey(0)
    return result


def getImageArea(src,debug):
    if debug:
        cv2.imshow("src", src)
        cv2.waitKey(0)

    # 颜色提取
    # 提取3种不同的颜色
    # 将图像转换为HSV颜色空间
    # 提取展区部分
    lower = np.array([90, 50, 100])
    upper = np.array([100, 255, 255])
    result_light_blue = get_color_mask(src, lower, upper, debug)
    # cv2.imwrite("./output/", result_light_blue)

    # 提取墙体部分
    lower = np.array([0, 30, 0])
    upper = np.array([90, 255, 255])
    result_green = get_color_mask(src, lower, upper, debug)
    # cv2.imwrite("./output/", result_green)

    # 提取地面部分
    lower = np.array([110, 110, 0])
    upper = np.array([140, 255, 200])
    result_dark_blue = get_color_mask(src, lower, upper, debug)
    # cv2.imwrite("./output/", result_dark_blue)
    images = {
        'exhibiitionArea': result_light_blue,
        'wall': result_green,
        'ground': result_dark_blue
    }
    return images


def getDist_P2L(PointP, Pointa, Pointb):
    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0] * Pointb[1] - Pointa[1] * Pointb[0]
    # 代入点到直线距离公式
    distance = (A * PointP[0] + B * PointP[1] + C) / math.sqrt(A * A + B * B)

    return distance


def getProjection_P2L(PointP, Pointa, Pointb):
    A = Pointa[1] - Pointb[1]
    B = Pointb[0] - Pointa[0]
    C = Pointa[0] * Pointb[1] - Pointa[1] * Pointb[0]

    x0 = (B ** 2 * PointP[0] - A * B * PointP[1] - A * C) / (A ** 2 + B ** 2)
    y0 = (A ** 2 * PointP[1] - A * B * PointP[0] - B * C) / (A ** 2 + B ** 2)
    projection_point = (x0, y0)

    return projection_point


def test_line(lines_new, x1, y1, x2, y2, threshold):
    for num_line, line in enumerate(lines_new):
        line_point1_x, line_point1_y, line_point2_x, line_point2_y = line
        line_point1 = (line_point1_x, line_point1_y)
        line_point2 = (line_point2_x, line_point2_y)
        distance1 = getDist_P2L([x1, y1], line_point1, line_point2)
        distance1 = abs(distance1)
        distance2 = getDist_P2L([x2, y2], line_point1, line_point2)
        distance2 = abs(distance2)
        if distance1 < threshold and distance2 < threshold:
            # 计算点在直线上的投影点，通过投影点计算确认直线情况
            projection_point1 = getProjection_P2L([x1, y1], line_point1, line_point2)
            projection_point2 = getProjection_P2L([x2, y2], line_point1, line_point2)
            if line_point1[0] > line_point2[0]:
                line_point1, line_point2 = line_point2, line_point1
            if projection_point1[0] > projection_point2[0]:
                projection_point1, projection_point2 = projection_point2, projection_point1

            # 包含
            if line_point1[0] <= projection_point1[0] and line_point2[0] >= projection_point2[0]:
                return False
            # 部分重叠或者相接
            if projection_point1[0] <= line_point1[0] <= projection_point2[0]:
                lines_new[num_line] = np.array(
                    [projection_point1[0], projection_point1[1], line_point2[0], line_point2[1]])
                return False
            if projection_point1[0] <= line_point2[0] <= projection_point2[0]:
                lines_new[num_line] = np.array(
                    [line_point1[0], line_point1[1], projection_point2[0], projection_point2[1]])
                return False
    return True


def get_edge(outPath, image, threshold, debug):
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 展示灰度图
    if debug:
        cv2.imshow("gray", gray)
        cv2.waitKey(0)

    # # Inverse_frame_gray = cv2.bitwise_not(gray)
    # Inverse_frame_gray = gray
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(Inverse_frame_gray)
    # # 去除连通域小的部分
    # testdelete = np.zeros((gray.shape[0], gray.shape[1]), np.uint8)  # 创建个全0的黑背景
    # for i in range(1, ret):
    #     mask = labels == i  # 这一步是通过labels确定区域位置，让labels信息赋给mask数组，再用mask数组做img数组的索引
    #     if stats[i][4] > 300:  # 300是面积 可以随便调
    #         testdelete[mask] = 255
    #         # 面积大于300的区域涂白留下，小于300的涂0抹去
    #     else:
    #         testdelete[mask] = 0
    # kernel = np.ones((5, 5), np.uint8)
    # testdelete = cv2.erode(testdelete, kernel, iterations=1)
    # # kernel = np.ones((3, 3), np.uint8)
    # # testdelete = cv2.dilate(testdelete, kernel, iterations=1)
    # if debug:
    #     cv2.imshow("delete", testdelete)
    #     output = cv2.bitwise_not(testdelete)
    #     cv2.imwrite("./output/test_delete.jpg",output)
    #     cv2.waitKey(0)

    # 骨架提取将图像中粗的直线转成细的
    # testdelete0 = testdelete.copy()
    testdelete0 = gray.copy()
    testdelete0[testdelete0 == 255] = 1
    skeleton0 = skeletonize(testdelete0)
    skeleton = skeleton0.astype(np.uint8) * 255
    if debug:
        cv2.imshow("skeleton", skeleton)
        kernel = np.ones((5, 5), np.uint8)
        skeleton2 = cv2.dilate(skeleton, kernel, iterations=1)
        output = cv2.bitwise_not(skeleton2)
        cv2.imwrite("./output/skeleton.jpg", output)
        cv2.waitKey(0)
    gray = skeleton

    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # if debug:
    #     cv2.imshow("skeleton gaussian", gray)
    #     cv2.waitKey(0)
    # 试一下直线拟合
    edge_output = cv2.Canny(gray, 50, 100)
    # 展示边缘提取的结果
    if debug:
        cv2.imshow("Canny Edge", edge_output)
        cv2.waitKey(0)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 5, minLineLength=10, maxLineGap=50)
    # img_RGB = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img_RGB2 = np.zeros(image.shape, dtype=np.uint8)
    print("the length of lines ", len(lines))
    lines_new = []
    for num_line, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        # cv2.line(img_RGB, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # if debug:
        #     print("the length of lines ", num_line, "x1,y1,x2,y2", x1, y1, x2, y2)
        #     cv2.imshow("lines", img_RGB)
        #     cv2.waitKey(0)
        if test_line(lines_new, x1, y1, x2, y2, threshold):
            # cv2.line(img_RGB, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(img_RGB2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lines_new.append(line[0])
            # if debug:
            #     print("the length of lines ", num_line, "x1,y1,x2,y2", x1, y1, x2, y2)
            #     cv2.imshow("lines", img_RGB)
            #     cv2.waitKey(0)
    if debug:
        cv2.imshow("lines", img_RGB2)
        # cv2.imshow("lines2", img_RGB)
        cv2.waitKey(0)

    # 从img_RGB2获取顶点
    # 图像连通域
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dist)

    # 迭代停止规则
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    # corners = cv2.goodFeaturesToTrack(gray, 75, 0.01, 10)
    # corners = np.int0(corners)
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(draw2, (x, y), 5, (0,0,255), -1)
    # 确定角点位置并绘制在draw2上
    for i in res:
        x1, y1, x2, y2 = i.ravel()
        corners2.append([x1, y1])
        cv2.circle(draw2, (x1, y1), 3, 255, -1)
    corners = corners2
    # 展示角点位置
    if debug:
        cv2.imshow("corners", draw2)
        cv2.imwrite("./output/corners.jpg",draw2)
        cv2.waitKey(0)
        print("the length of corners", len(corners))

    # 为角点添加序号标号方便后续调整位置后查找
    for i in range(len(corners)):
        corners[i].append(i)
    print(lines_new)


    # Inverse_skeleton = cv2.bitwise_not(skeleton)
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton)
    # # 显示连通域识别的结果
    # output = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    # for i in range(1, ret):
    #     mask = labels == i
    #     output[:, :, 0][mask] = np.random.randint(0, 255)
    #     output[:, :, 1][mask] = np.random.randint(0, 255)
    #     output[:, :, 2][mask] = np.random.randint(0, 255)
    # if debug:
    #     cv2.imshow("delete2", output)
    #     cv2.waitKey(0)

    # # 边缘提取
    # edge_output = cv2.Canny(gray, 50, 100)
    # # 展示边缘提取的结果
    # if debug:
    #     cv2.imshow("Canny Edge", edge_output)
    #     cv2.waitKey(0)

    # skeleton2 = cv2.bitwise_not(skeleton2)
    # draw2 = cv2.cvtColor(skeleton2, cv2.COLOR_GRAY2RGB)
    # # 注意：需要copy一下，否则将会改动原图
    # # 获取角点
    # dist = cv2.cornerHarris(gray, blockSize=5, ksize=7, k=0.05)
    # # dist = cv2.goodFeaturesToTrack(gray, 75, 0.01, 10)
    # ret, dist = cv2.threshold(dist, 0.005 * dist.max(), 255, 0)
    # dist = np.uint8(dist)
    # corners2 = []
    #
    # # 图像连通域
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dist)
    #
    # # 迭代停止规则
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # res = np.hstack((centroids, corners))
    # res = np.int0(res)
    # # corners = cv2.goodFeaturesToTrack(gray, 75, 0.01, 10)
    # # corners = np.int0(corners)
    # # for i in corners:
    # #     x, y = i.ravel()
    # #     cv2.circle(draw2, (x, y), 5, (0,0,255), -1)
    # # 确定角点位置并绘制在draw2上
    # for i in res:
    #     x1, y1, x2, y2 = i.ravel()
    #     corners2.append([x1, y1])
    #     cv2.circle(draw2, (x1, y1), 3, 255, -1)
    # corners = corners2
    # # 展示角点位置
    # if debug:
    #     cv2.imshow("corners", draw2)
    #     cv2.imwrite("./output/corners.jpg",draw2)
    #     cv2.waitKey(0)
    #     print("the length of corners", len(corners))
    #
    # # 为角点添加序号标号方便后续调整位置后查找
    # for i in range(len(corners)):
    #     corners[i].append(i)
    # orginCorners = corners
    # corners = sorted(corners, key=itemgetter(0, 1))
    # # 确定水平和竖直的列表
    # verticalCornerList = []
    # index = 0
    # while index < len(corners):
    #     cornerX = corners[index][0]
    #     nextCorner = index + 1
    #     verticalCorner = [corners[index]]
    #     while nextCorner < len(corners) and abs(corners[nextCorner][0] - cornerX) < threshold:
    #         corners[nextCorner][0] = cornerX
    #         verticalCorner.append(corners[nextCorner])
    #         nextCorner += 1
    #     verticalCornerList.append(verticalCorner)
    #     index = nextCorner
    #
    # corners = sorted(corners, key=itemgetter(1, 0))
    # horizontalCornerList = []
    # index = 0
    # while index < len(corners):
    #     cornerY = corners[index][1]
    #     nextCorner = index + 1
    #     horizontalCorner = [corners[index]]
    #     while nextCorner < len(corners) and abs(corners[nextCorner][1] - cornerY) < threshold:
    #         corners[nextCorner][1] = cornerY
    #         horizontalCorner.append(corners[nextCorner])
    #         nextCorner += 1
    #     horizontalCornerList.append(horizontalCorner)
    #     index = nextCorner
    #
    # # 去除完全相同的点
    # for cornerList in verticalCornerList:
    #     for cornerIndex in range(len(cornerList)):
    #         for cornerIndex2 in range(len(cornerList)):
    #             if cornerIndex2 != cornerIndex and cornerList[cornerIndex][2] != -1 and cornerList[cornerIndex2][
    #                 2] != -1:
    #                 if cornerList[cornerIndex][0] == cornerList[cornerIndex2][0] and cornerList[cornerIndex][1] == \
    #                         cornerList[cornerIndex2][1]:
    #                     cornerList[cornerIndex][2] = -1
    #
    # # 给水平和垂直的列表排序
    # for index, cornerList in enumerate(verticalCornerList):
    #     verticalCornerList[index] = sorted(cornerList, key=itemgetter(1))
    # for index, cornerList in enumerate(horizontalCornerList):
    #     horizontalCornerList[index] = sorted(cornerList, key=itemgetter(0))
    #
    # walls = []
    # # 四个方向按上下左右计算
    # neighborCorner = np.zeros((len(corners), 4), dtype=int)
    # # 对应neighborCorner存储上下左右四个方向的顶点在水平，竖直列表中的位置方便后续合并墙体
    # neighborIndex = np.empty((len(corners), 4, 2), dtype=int)
    # # 检测拐点之间是否存在墙
    # for cornerListIndex, cornerList in enumerate(verticalCornerList):
    #     if len(cornerList) >= 2:
    #         UpCorner = 0
    #         DownCorner = 1
    #         while DownCorner < len(cornerList) and cornerList[UpCorner][2] == -1:
    #             UpCorner = DownCorner
    #             DownCorner += 1
    #         while DownCorner < len(cornerList):
    #             if cornerList[DownCorner][2] == -1:
    #                 DownCorner += 1
    #                 continue
    #             # 对中点进行检测(这个检测方法有问题)
    #             # 截取两点之间的图片然后用轮廓提取获取线段位置
    #             # if cornerList[DownCorner][1]-cornerList[UpCorner][1]-threshold*2>0:
    #             #     test = edge_output[cornerList[UpCorner][1]+threshold:cornerList[DownCorner][1]-threshold,
    #             #        max(cornerList[UpCorner][0]-threshold, 0):cornerList[UpCorner][0]+threshold]
    #             #     height, width = test.shape
    #             # else:
    #             test = gray[cornerList[UpCorner][1]:cornerList[DownCorner][1],
    #                    max(cornerList[UpCorner][0] - threshold, 0):cornerList[UpCorner][0] + threshold]
    #             height, width = test.shape
    #             contours, hierarchy = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             # if debug:
    #             #     test1 = test.copy()
    #             #     cv2.drawContours(test1, contours, -1, (255, 255, 255), 1)
    #             #     cv2.imshow("contours_test", test1)
    #             #     cv2.waitKey(0)
    #             wall_exist = False
    #             for contour in contours:
    #                 if len(contour) == 2:
    #                     if abs(contour[0][0][0] - contour[1][0][0]) < threshold and \
    #                             abs(abs(contour[0][0][1] - contour[1][0][1]) - height) < threshold:
    #                         wall_exist = True
    #                         break
    #                 elif len(contour) > 2:
    #                     for point1 in range(len(contour)):
    #                         for point2 in range(point1 + 1, len(contour)):
    #                             if abs(contour[point1][0][0] - contour[point2][0][0]) < threshold and \
    #                                     abs(abs(contour[point1][0][1] - contour[point2][0][1]) - height) < threshold:
    #                                 wall_exist = True
    #                                 break
    #             # center = (cornerList[UpCorner][1]+cornerList[DownCorner][1])//2
    #             # test = edge_output[center-threshold:center+threshold,
    #             #        max(cornerList[UpCorner][0]-threshold,0):cornerList[UpCorner][0]+threshold]
    #             # 显示截取的图的样子
    #             # if debug:
    #             #     cv2.imshow("test",test)
    #             #     cv2.waitKey(0)
    #             # 检测截取的部分是否有非零位置
    #             # List = np.argwhere(test)
    #             # 添加墙
    #             if wall_exist:
    #                 neighborCorner[cornerList[UpCorner][2]][1] = 1
    #                 neighborCorner[cornerList[DownCorner][2]][0] = 1
    #                 neighborIndex[cornerList[UpCorner][2]][1][0] = cornerListIndex
    #                 neighborIndex[cornerList[UpCorner][2]][1][1] = DownCorner
    #                 neighborIndex[cornerList[DownCorner][2]][0][0] = cornerListIndex
    #                 neighborIndex[cornerList[DownCorner][2]][0][1] = UpCorner
    #                 walls.append([cornerList[UpCorner], cornerList[DownCorner]])
    #             UpCorner = DownCorner
    #             DownCorner += 1
    #
    # for cornerListIndex, cornerList in enumerate(horizontalCornerList):
    #     if len(cornerList) >= 2:
    #         LeftCorner = 0
    #         RightCorner = 1
    #         while RightCorner < len(cornerList) and cornerList[LeftCorner][2] == -1:
    #             LeftCorner = RightCorner
    #             RightCorner += 1
    #         while RightCorner < len(cornerList):
    #             if cornerList[RightCorner][2] == -1:
    #                 # LeftCorner = RightCorner
    #                 RightCorner += 1
    #                 continue
    #             # center = (cornerList[LeftCorner][0] + cornerList[RightCorner][0]) // 2
    #             # test = edge_output[max(cornerList[LeftCorner][1]-threshold,0):cornerList[LeftCorner][1]+threshold,
    #             # center - threshold:center + threshold]
    #             # List = np.argwhere(test)
    #             # 截取两点之间的图片然后用轮廓提取获取线段位置
    #             # if cornerList[RightCorner][0] - cornerList[LeftCorner][0] - threshold * 2 > 0:
    #             #     test = edge_output[max(cornerList[LeftCorner][1]-threshold,0):cornerList[LeftCorner][1]+threshold,
    #             #             cornerList[LeftCorner][0] + threshold:cornerList[RightCorner][0] - threshold]
    #             #     height, width = test.shape
    #             # else:
    #             test = gray[max(cornerList[LeftCorner][1] - threshold, 0):cornerList[LeftCorner][1] + threshold,
    #                    cornerList[LeftCorner][0]:cornerList[RightCorner][0]]
    #             height, width = test.shape
    #             contours, hierarchy = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #             # if debug:
    #             #     test1 = test.copy()
    #             #     cv2.drawContours(test1, contours, -1, (255, 255, 255), 1)
    #             #     cv2.imshow("contours_test", test1)
    #             #     cv2.waitKey(0)
    #             wall_exist = False
    #             for contour in contours:
    #                 if len(contour) == 2:
    #                     if abs(contour[0][0][1] - contour[1][0][1]) < threshold and \
    #                             abs(abs(contour[0][0][0] - contour[1][0][0]) - width) < threshold:
    #                         wall_exist = True
    #                         break
    #                 elif len(contour) > 2:
    #                     for point1 in range(len(contour)):
    #                         for point2 in range(point1 + 1, len(contour)):
    #                             if abs(contour[point1][0][1] - contour[point2][0][1]) < threshold and \
    #                                     abs(abs(contour[point1][0][0] - contour[point2][0][0]) - width) < threshold:
    #                                 wall_exist = True
    #                                 break
    #             # 显示截取的图的样子
    #             # if debug:
    #             #     cv2.imshow("test", test)
    #             #     cv2.waitKey(0)
    #             if wall_exist:
    #                 neighborCorner[cornerList[LeftCorner][2]][3] = 1
    #                 neighborCorner[cornerList[RightCorner][2]][2] = 1
    #                 neighborIndex[cornerList[LeftCorner][2]][3][0] = cornerListIndex
    #                 neighborIndex[cornerList[LeftCorner][2]][3][1] = RightCorner
    #                 neighborIndex[cornerList[RightCorner][2]][2][0] = cornerListIndex
    #                 neighborIndex[cornerList[RightCorner][2]][2][1] = LeftCorner
    #                 walls.append([cornerList[LeftCorner], cornerList[RightCorner]])
    #             LeftCorner = RightCorner
    #             RightCorner += 1
    #
    # # 筛选只有上下或者只有左右墙的点进行合并
    # for index, neighbor in enumerate(neighborCorner):
    #     # 上下
    #     if neighbor[0] == 1 and neighbor[1] == 1 and neighbor[2] == 0 and neighbor[3] == 0:
    #         # 去除该点
    #         orginCorners[index][2] = -1
    #         # 合并墙壁
    #         neighbor[0] = 0
    #         neighbor[1] = 0
    #         verticalIndex = neighborIndex[index][0][0]
    #         UpCorner = neighborIndex[index][0][1]
    #         DownCorner = neighborIndex[index][1][1]
    #         neighborIndex[verticalCornerList[verticalIndex][UpCorner][2]][1][1] = DownCorner
    #         neighborIndex[verticalCornerList[verticalIndex][DownCorner][2]][0][1] = UpCorner
    #         walls.append([verticalCornerList[verticalIndex][UpCorner], verticalCornerList[verticalIndex][DownCorner]])
    #     # 左右
    #     if neighbor[0] == 0 and neighbor[1] == 0 and neighbor[2] == 1 and neighbor[3] == 1:
    #         orginCorners[index][2] = -1
    #         neighbor[2] = 0
    #         neighbor[3] = 0
    #         horizontalIndex = neighborIndex[index][2][0]
    #         LeftCorner = neighborIndex[index][2][1]
    #         RightCorner = neighborIndex[index][3][1]
    #         neighborIndex[horizontalCornerList[horizontalIndex][LeftCorner][2]][3][1] = RightCorner
    #         neighborIndex[horizontalCornerList[horizontalIndex][RightCorner][2]][2][1] = LeftCorner
    #         walls.append(
    #             [horizontalCornerList[horizontalIndex][LeftCorner], horizontalCornerList[horizontalIndex][RightCorner]])
    #     # 仅有一个方向有墙的拐点
    #     # sum = np.count_nonzero(neighbor)
    #     # if sum == 1:
    #     #     if neighbor[0]==1 or neighbor[1]==1:
    #
    # # 去除-1为顶点的walls的内容
    # newWalls = []
    # for wall in walls:
    #     if wall[0][2] != -1 and wall[1][2] != -1:
    #         newWalls.append(wall)
    # walls = newWalls
    #
    # # 查看walls的结果
    # width, height = gray.shape
    # wallsImg = np.zeros((width, height, 3), np.uint8)  # 生成一个空灰度图像
    # color = 1
    # for wall in walls:
    #     color = color + 2
    #     cv2.line(wallsImg, (orginCorners[wall[0][2]][0], orginCorners[wall[0][2]][1]),
    #              (orginCorners[wall[1][2]][0], orginCorners[wall[1][2]][1]), (0, 0, 255), 5, 4)
    # cv2.imwrite(outPath + "walls.png", wallsImg)
    # if debug:
    #     cv2.imshow('walls', wallsImg)
    # return orginCorners, walls


def base64_to_cv2(base64_code):
    img_data = base64.b64decode(base64_code)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return img

@app.route('/imagechange', methods=['POST'])
def data():
    InputData = request.get_json()
    print(InputData['image'])
    image = base64_to_cv2(InputData['image'])
    images = getImageArea(image,True)
    corners, walls = get_edge("./output/",images['wall'],20,True)
    return "OK"



if __name__ == '__main__':
    app.run()

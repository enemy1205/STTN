import os
import numpy as np
import cv2

T_H = 64
T_W = 64

def get_location(img):
    if img.sum() <= 30*30*255:  # 没有人体轮廓
        return None, None, None, None

    y = img.sum(axis=1)  # 按行求和，结果为列向量（长度240）
    y_top = (y != 0).argmax(axis=0)  # 返回列向量y中非零值索引的最大值（不是最大值的索引）（即轮廓头部）
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)  # 返回列向量y中非零值累计求和最大值的索引（即轮廓底部
    if (y_btm-y_top) < 50:
        return None, None, None, None
    img = img[y_top:y_btm + 1, :]  # 77*320 取出img人体轮廓头与脚之间的图像
    _r = img.shape[1] / img.shape[0]  # 计算取出图像的宽高比
    _t_w = int(T_H * _r)  # 计算按宽高比得到的处理后图片的宽度
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)  # 等比缩放
    # # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()  # 所有元素求和
    sum_column = img.sum(axis=0).cumsum()  # 按列求和并累计再求和
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i  # 人体轮廓中心横坐标
            break
    if x_center < 0:  # 没有人体轮廓
        return None, None, None, None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W  # 人体轮廓最左侧横坐标
    right = x_center + h_T_W  # 人体轮廓最右侧横坐标
    if left <= 0 or right >= img.shape[1]:
        return None, None, None, None
    return y_top, y_btm, left, right


def cut_img(img, y_top, y_btm, left, right):
    img = img[y_top:y_btm + 1, :]  # 取出img人体轮廓头与脚之间的图像
    _r = img.shape[1] / img.shape[0]  # 计算取出图像的宽高比
    _t_w = int(T_H * _r)  # 计算按宽高比得到的处理后图片的宽度
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)  # 等比缩放
    h_T_W = int(T_W / 2)
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]  # 取出64*64的人体轮廓
    return img.astype('uint8')

# def cut_img(img):
#     if img.sum() <= 10000:  # 没有人体轮廓
#         return None
#     y = img.sum(axis=1)  # 按行求和，结果为列向量（长度240）
#     y_top = (y != 0).argmax(axis=0)  # 返回列向量y中非零值索引的最大值（不是最大值的索引）（即轮廓头部）
#     y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)  # 返回列向量y中非零值累计求和最大值的索引（即轮廓底部）
#     img = img[y_top:y_btm + 1, :]  # 77*320 取出img人体轮廓头与脚之间的图像
#     _r = img.shape[1] / img.shape[0]  # 计算取出图像的宽高比
#     _t_w = int(T_H * _r)  # 计算按宽高比得到的处理后图片的宽度
#     img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)  # 等比缩放
#     # Get the median of x axis and regard it as the x center of the person.
#     sum_point = img.sum()  # 所有元素求和
#     sum_column = img.sum(axis=0).cumsum()  # 按列求和并累计再求和
#     x_center = -1
#     for i in range(sum_column.size):
#         if sum_column[i] > sum_point / 2:
#             x_center = i  # 人体轮廓中心横坐标
#             break
#     if x_center < 0:  # 没有人体轮廓
#         return None
#     h_T_W = int(T_W / 2)
#     left = x_center - h_T_W  # 人体轮廓最左侧横坐标
#     right = x_center + h_T_W  # 人体轮廓最右侧横坐标
#     if left <= 0 or right >= img.shape[1]:
#         left += h_T_W
#         right += h_T_W
#         _ = np.zeros((img.shape[0], h_T_W))
#         img = np.concatenate([_, img, _], axis=1)
#     img = img[:, left:right]  # 取出64*64的人体轮廓
#     return img.astype('uint8')


if __name__ == "__main__":
    img_path = "/home/fh/mupeg-master/temp/silhouettes_two_person"
    img_save_path = "/home/fh/mupeg-master/temp/silhouettes_two_person_cut"
    img_list = os.listdir(img_path)
    img_list.sort()
    for _img in img_list:
        img = os.path.join(img_path, _img)
        image = cv2.imread(img, 0)
        y_top, y_btm, left, right = get_location(image.copy())
        img_cut = cut_img(image, y_top, y_btm, left, right)
        img_save = os.path.join(img_save_path, _img)
        cv2.imwrite(img_save, img_cut)
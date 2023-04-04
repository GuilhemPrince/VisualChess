import os
import matplotlib.pyplot as plt
import cv2
print(os.getcwd())

def get_list_img_path(folder_path):
    elements = sorted(os.listdir(folder_path))
    imgs_path = [folder_path + img_name for img_name in elements]
    return imgs_path

def show_and_get_gray_img(path):
    img_gray = cv2.imread(path,0)
    plt.imshow(img_gray, cmap='gray')
    plt.show()
    return img_gray

def show_and_get_colored_img(path):
    img_bgr = cv2.imread(path,1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    return img_rgb

if __name__ == '__main__':
    show_and_get_colored_img('.././photos_test/square/with_pieces/black_on_white_g.png')
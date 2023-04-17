import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
print(os.getcwd())

############        General            ############

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

def get_test_boxes():
    boxes = {'a1': [(97, 68), (163, 67), (95, 139), (162, 138)], 
         'a2': [(163, 67), (235, 66), (162, 138), (235, 137)], 
         'a3': [(235, 66), (307, 65), (235, 137), (307, 136)], 
         'a4': [(307, 65), (378, 64), (307, 136), (379, 134)], 
         'a5': [(378, 64), (446, 64), (379, 134), (448, 133)], 
         'a6': [(446, 64), (514, 62), (448, 133), (517, 132)], 
         'a7': [(514, 62), (580, 60), (517, 132), (584, 131)], 
         'a8': [(580, 60), (655, 59), (584, 131), (660, 130)], 
         'b1': [(95, 139), (162, 138), (93, 207), (161, 206)], 
         'b2': [(162, 138), (235, 137), (161, 206), (235, 205)], 
         'b3': [(235, 137), (307, 136), (235, 205), (307, 204)], 
         'b4': [(307, 136), (379, 134), (307, 204), (380, 203)], 
         'b5': [(379, 134), (448, 133), (380, 203), (450, 202)], 
         'b6': [(448, 133), (517, 132), (450, 202), (520, 199)], 
         'b7': [(517, 132), (584, 131), (520, 199), (589, 199)], 
         'b8': [(584, 131), (660, 130), (589, 199), (665, 198)], 
         'c1': [(93, 207), (161, 206), (91, 277), (160, 275)], 
         'c2': [(161, 206), (235, 205), (160, 275), (235, 274)], 
         'c3': [(235, 205), (307, 204), (235, 274), (307, 273)], 
         'c4': [(307, 204), (380, 203), (307, 273), (381, 272)], 
         'c5': [(380, 203), (450, 202), (381, 272), (452, 271)], 
         'c6': [(450, 202), (520, 199), (452, 271), (524, 268)], 
         'c7': [(520, 199), (589, 199), (524, 268), (593, 268)], 
         'c8': [(589, 199), (665, 198), (593, 268), (671, 267)], 
         'd1': [(91, 277), (160, 275), (89, 348), (159, 346)], 
         'd2': [(160, 275), (235, 274), (159, 346), (235, 345)], 
         'd3': [(235, 274), (307, 273), (235, 345), (307, 344)], 
         'd4': [(307, 273), (381, 272), (307, 344), (382, 343)], 
         'd5': [(381, 272), (452, 271), (382, 343), (454, 342)], 
         'd6': [(452, 271), (524, 268), (454, 342), (528, 339)], 
         'd7': [(524, 268), (593, 268), (528, 339), (598, 340)], 
         'd8': [(593, 268), (671, 267), (598, 340), (676, 338)], 
         'e1': [(89, 348), (159, 346), (86, 415), (158, 414)], 
         'e2': [(159, 346), (235, 345), (158, 414), (235, 412)], 
         'e3': [(235, 345), (307, 344), (235, 412), (307, 411)], 
         'e4': [(307, 344), (382, 343), (307, 411), (383, 410)], 
         'e5': [(382, 343), (454, 342), (383, 410), (457, 406)], 
         'e6': [(454, 342), (528, 339), (457, 406), (532, 408)], 
         'e7': [(528, 339), (598, 340), (532, 408), (602, 407)], 
         'e8': [(598, 340), (676, 338), (602, 407), (681, 406)], 
         'f1': [(86, 415), (158, 414), (84, 490), (157, 489)], 
         'f2': [(158, 414), (235, 412), (157, 489), (235, 487)], 
         'f3': [(235, 412), (307, 411), (235, 487), (307, 486)], 
         'f4': [(307, 411), (383, 410), (307, 486), (384, 485)], 
         'f5': [(383, 410), (457, 406), (384, 485), (459, 484)], 
         'f6': [(457, 406), (532, 408), (459, 484), (536, 483)], 
         'f7': [(532, 408), (602, 407), (536, 483), (607, 482)], 
         'f8': [(602, 407), (681, 406), (607, 482), (687, 481)], 
         'g1': [(84, 490), (157, 489), (82, 569), (156, 568)], 
         'g2': [(157, 489), (235, 487), (156, 568), (235, 566)], 
         'g3': [(235, 487), (307, 486), (235, 566), (307, 565)], 
         'g4': [(307, 486), (384, 485), (307, 565), (386, 564)], 
         'g5': [(384, 485), (459, 484), (386, 564), (462, 563)], 
         'g6': [(459, 484), (536, 483), (462, 563), (540, 562)], 
         'g7': [(536, 483), (607, 482), (540, 562), (612, 566)], 
         'g8': [(607, 482), (687, 481), (612, 566), (693, 556)], 
         'h1': [(82, 569), (156, 568), (80, 659), (154, 655)], 
         'h2': [(156, 568), (235, 566), (154, 655), (235, 653)], 
         'h3': [(235, 566), (307, 565), (235, 653), (307, 652)], 
         'h4': [(307, 565), (386, 564), (307, 652), (387, 651)], 
         'h5': [(386, 564), (462, 563), (387, 651), (465, 650)], 
         'h6': [(462, 563), (540, 562), (465, 650), (545, 649)], 
         'h7': [(540, 562), (612, 566), (545, 649), (617, 648)], 
         'h8': [(612, 566), (693, 556), (617, 648), (700, 647)]}
    return boxes

############         Piece detection gray         ############

def get_gradient(img, croop = 5):
    X, Y = img.shape
    img =  img[int(X/croop):int((croop-1)*X/croop), int(Y/croop):int((croop-1)*Y/croop)]
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)

    mag = np.sqrt(gx**2 + gy**2)
    mean_gradient = np.mean(mag)

    #print("Le gradient moyen de l'image est : ", round(mean_gradient, 1))
    return mean_gradient

def is_piece_in_square(img, treshold=7.5):
    return get_gradient(img) > treshold

############         Piece detection color         ############

def get_gradient_color(img):
    r_grad = get_gradient(img[:,:,0])
    g_grad = get_gradient(img[:,:,1])
    b_grad = get_gradient(img[:,:,2]) 
    mean_grad =    (r_grad + g_grad + b_grad) /3
    print("Le gradient moyen de l'image est : ", round(mean_grad, 1))
    return mean_grad

def is_piece_in_color_square(img, treshold=8):
    return get_gradient_color(img) > treshold


############         Color detection          ############

def get_central_part_of_square(img):
    X, Y = img.shape
    return img[int(X/4):int(3*X/4), int(Y/4):int(3*Y/4)]

def is_piece_white(img, treshold = 100 ):
    central_img = get_central_part_of_square(img)
    whiteness = np.mean(central_img)
    return whiteness > treshold

############         Square detection          ############


def get_coord_square(point_list):
    Xmin = int((point_list[0][0] + point_list[2][0]) / 2)
    Xmax = int((point_list[1][0] + point_list[3][0]) / 2)
    Ymin = int((point_list[0][1] + point_list[1][1]) / 2)
    Ymax = int((point_list[2][1] + point_list[3][1]) / 2)
    return Xmin, Xmax, Ymin, Ymax

def get_square(img, Xmin, Xmax, Ymin, Ymax):
    return img[Ymin:Ymax, Xmin:Xmax] # X et Y sont inversés, je ne sais pas pq


def get_all_squares(boxes, board):
    square_list = []
    for square in boxes:
        Xmin, Xmax, Ymin, Ymax = get_coord_square(boxes[square])
        square_list.append(get_square(board, Xmin, Xmax, Ymin, Ymax))
    return square_list

def square_color(square):
    '''
        return 0 if the square is empty
        return 1 if there is a white piece in the square
        return 2 if the piece is black
    '''
    if is_piece_in_square(square):
        return 2 - is_piece_white(square) 
    return 0

def squares_to_array(square_list):
    board = np.zeros([8,8])
    for i in range(8):
        for j in range(8):
            board[i,j] = square_color(square_list[j * 8 + (7-i)])
    return board

############         Array Modeliastion          ############

def get_initial_position():
    initial_position = np.array([['R','N','B','K','Q','B','N','R'], 
                             ['P']*8, 
                             [' ']*8, 
                             [' ']*8, 
                             [' ']*8, 
                             [' ']*8, 
                             ['p']*8, 
                             ['r','n','b','k','q','b','n','r']])
    return initial_position

def get_square_type(pos_square):
    if pos_square == ' ':
        return 0
    if pos_square.lower() == pos_square: # les blancs sont en lower_case
        return 1
    return 2

def is_final_square(move_square, pos_square):
    if move_square == 0:
        return False
    if get_square_type(pos_square) == move_square:
        return False
    return True

def is_capture(final_pos_square):
    return final_pos_square != ' '

def get_queen(final_pos_square):
    if final_pos_square.lower() == final_pos_square: 
        return 'q'
    return 'Q'

def print_move(pos_square, coord, capture, queening):
    piece = ''
    capt = ''
    queen = ''
    if pos_square.lower() != 'p':
        piece = pos_square.upper()
    if capture :
        capt = 'x'
    if queening :
        queen = '-Q'
    column = chr(coord[0] + 97)
    line = coord[1] + 1
    print(piece + capt + str(column) + str(line) + queen)

def new_position(current_position, new_move):
    original_square = [9,9]
    final_square = [9,9]
    capture = False
    for c in range(8):
        for l in range(8):
            move_square = new_move[c,l]
            pos_square = current_position[c,l]
            if move_square == 0 and pos_square != ' ' :
                original_square = (c,l)
            if is_final_square(move_square, pos_square):
                final_square = (c,l)
                capture = is_capture(pos_square)
    current_position[final_square] = current_position[original_square]
    current_position[original_square] = ' '
    # Is a player queening ?
    queening = current_position[final_square].lower() == 'p' and (final_square[0] == 0 or final_square[0] == 7)

    # Print the Move
    print_move(current_position[final_square], final_square, capture, queening)

    # If the player is queening, replace the pawn by the queen
    if queening :
        current_position[final_square] = get_queen(current_position[final_square])

    return current_position
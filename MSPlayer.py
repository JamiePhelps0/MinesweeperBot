import time
import tkinter as tk
import keyboard
import tensorflow as tf
import mouse
from PIL import ImageGrab
from PIL import Image
import numpy as np
np.set_printoptions(linewidth=400)


def clear():
    global viewing_board
    viewing_board = [-1 for _ in range(480)]


def encode():
    state = np.array(viewing_board, dtype=np.float32).reshape((480,))
    return np.array([[0 if i == -1 else 1, 0.125 * (i + 1)] for i in state], dtype=np.float32).reshape((1, 16, 30, 2))


def find_move():
    # action = np.argmax(net(encode()))
    next_val = np.array(net(encode()))[0]
    actions = np.where(next_val >= 0.97)[0]
    print(actions)
    width = r_corner[0] - l_corner[0]
    height = r_corner[1] - l_corner[1]
    w_pix = width / 30
    h_pix = height / 16
    for i in actions:
        pos = (l_corner[0] + w_pix / 2 + w_pix * (i % 30), l_corner[1] + h_pix / 2 + h_pix * (i // 30))
        mouse.move(pos[0], pos[1])
        mouse.click()
    get_state()
    next_val = np.array(net(encode()))[0]
    print(np.round(next_val, 3).reshape((16, 30)))
    print(np.where(next_val >= 0.9))
    print('*' * 20)
    find_move_label.config(text=f'{action % 30}, {action // 30}')


def get_state():
    global viewing_board
    if (r_corner == (0, 0)) or (l_corner == (0, 0)):
        print('record corner positions')
        return
    screen = ImageGrab.grab((l_corner[0], l_corner[1], r_corner[0], r_corner[1]))
    screen = np.array(screen)
    screen = np.array(tf.image.rgb_to_grayscale(tf.image.resize(screen, (480, 900))))
    # screen = np.array(tf.image.resize(screen, (480, 900)))
    units = np.array([[screen[i * 30:(i + 1) * 30, j * 30: (j + 1) * 30] for i in range(16)] for j in range(30)])
    game_state = []
    for row in range(16):
        for col in range(30):
            max_psnr = 0
            best_fit = None
            for key in reference_images.keys():
                mse = np.mean(np.abs(reference_images[key] - units[col, row]))
                psnr = 10 * np.log10(255 ** 2 / mse)
                if psnr > max_psnr:
                    max_psnr = psnr
                    best_fit = key
            game_state.append(best_fit)
    [item.config(image=img_dict[key]) for key, item in zip(game_state, board)]
    viewing_board = [int(i) if i != 'unknown' else -1 for i in game_state]


def auto_play():
    width = r_corner[0] - l_corner[0]
    height = r_corner[1] - l_corner[1]
    w_pix = width / 30
    h_pix = height / 16
    for _ in range(70):
        next_val = np.array(net(encode()))[0]
        # actions = np.where(next_val >= 0.99)[0]
        actions = []
        print(np.round(next_val, 3).reshape((16, 30)))
        print(actions)
        print('*' * 20)
        if len(actions) == 0:
            actions = [np.argmax(next_val)]
        for i in actions:
            pos = (l_corner[0] + w_pix / 2 + w_pix * (i % 30), l_corner[1] + h_pix / 2 + h_pix * (i // 30))
            mouse.move(pos[0], pos[1])
            mouse.click()
        get_state()


def set_l_corner():
    global l_corner
    while not keyboard.is_pressed('enter'):
        time.sleep(0.01)
    l_pos = mouse.get_position()
    l_corner_button.config(text=f'{l_pos[0]}, {l_pos[1]}')
    l_corner = l_pos


def set_r_corner():
    global r_corner
    while not keyboard.is_pressed('enter'):
        time.sleep(0.05)
    r_pos = mouse.get_position()
    r_corner_button.config(text=f'{r_pos[0]}, {r_pos[1]}')
    r_corner = r_pos


root = tk.Tk()
root.title('Minesweeper Assistant')
root.geometry('1500x600+50+50')
root.resizable(False, False)
root.configure(bg='black')


def get_image_array(directory):
    image = Image.open(f'./assets/{directory}')
    image = np.array(image)[:, :, :3]
    image = np.array(tf.image.rgb_to_grayscale(image))
    return image


reference_images = {'0': get_image_array('0.PNG'),
                    '1': get_image_array('1.PNG'),
                    '2': get_image_array('2.PNG'),
                    '3': get_image_array('3.PNG'),
                    '4': get_image_array('4.PNG'),
                    '5': get_image_array('5.PNG'),
                    '6': get_image_array('6.PNG'),
                    '7': get_image_array('7.PNG'),
                    'unknown': get_image_array('unknown.PNG')}

r_corner = (0, 0)
l_corner = (0, 0)

net = tf.keras.models.load_model('Saves/DRN_xbig_s.h5')

init_state = [-1 for _ in range(480)]

img_dict = {'0': tk.PhotoImage(file='./assets/0.PNG'),
            '1': tk.PhotoImage(file='./assets/1.PNG'),
            '2': tk.PhotoImage(file='./assets/2.PNG'),
            '3': tk.PhotoImage(file='./assets/3.PNG'),
            '4': tk.PhotoImage(file='./assets/4.PNG'),
            '5': tk.PhotoImage(file='./assets/5.PNG'),
            '6': tk.PhotoImage(file='./assets/6.PNG'),
            '7': tk.PhotoImage(file='./assets/7.PNG'),
            'unknown': tk.PhotoImage(file='./assets/unknown.PNG')}

viewing_board = [-1 for _ in range(480)]

net(encode())

board_frame = tk.Frame(root)
board = [tk.Label(board_frame, image=img_dict['unknown'], borderwidth=1) for i in range(480)]
[item.grid(column=i % 30, row=i // 30) for i, item in enumerate(board)]

board_frame.place(x=40, y=40)

y_labels = [tk.Label(root, text=str(i)) for i in range(16)]
[item.place(y=46 + i * 32, x=15, width=20, height=20) for i, item in enumerate(y_labels)]

x_labels = [tk.Label(root, text=str(i)) for i in range(30)]
[item.place(y=10, x=44 + i * 32, width=20, height=20) for i, item in enumerate(x_labels)]

find_move_button = tk.Button(root, text='Find Best Action', command=find_move)
find_move_button.place(x=1100, y=120, width=150, height=50)
find_move_label = tk.Label(root)
find_move_label.place(x=1250, y=120, width=150, height=50)

get_state_button = tk.Button(root, text='Get State', command=get_state)
get_state_button.place(x=1100, y=250, width=195, height=50)

auto_complete = tk.Button(root, text='Auto-Play Game', command=auto_play)
auto_complete.place(x=905, y=250, width=195, height=50)

delay_label = tk.Label(root, text='Action Delay for Auto-Play')
delay_label.place(x=920, y=200, width=50, height=20)
delay_entry = tk.Entry(root)
delay_entry.place(x=975, y=200, width=50, height=20)

l_corner_label = tk.Label(root, text='L-Corner')
l_corner_label.place(x=920, y=300, width=50, height=20)
l_corner_button = tk.Button(root, text=f'{l_corner[0]}, {l_corner[1]}', command=set_l_corner)
l_corner_button.place(x=975, y=300, width=70, height=20)

r_corner_label = tk.Label(root, text='R-Corner')
r_corner_label.place(x=920, y=350, width=50, height=20)
r_corner_button = tk.Button(root, text=f'{r_corner[0]}, {r_corner[1]}', command=set_r_corner)
r_corner_button.place(x=975, y=350, width=70, height=20)

clear_button = tk.Button(root, text='clear', command=clear)
clear_button.place(x=1100, y=400, width=70, height=20)

root.mainloop()

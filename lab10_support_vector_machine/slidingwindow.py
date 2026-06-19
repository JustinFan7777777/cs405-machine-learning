import numpy as np
import matplotlib.pyplot as plt
import cv2

# image_size = (1280, 720): the width & height of the input image
# init_size = (64, 64): the initial size of the sliding window at the bottom of the image (y_range[0])
# x_overlap = 0.5: meaning that adjacent windows at a given y will overlap by 50% in the x direction
# y_step = 0.05: the vertical step size -> each time move up by 5% of the image height (along y direction)
# x_range = (0, 1): horizontal search area -> across entire width from left -> right
# y_range = (0, 1): vertical search area -> across entire height from top -> bottom

# scale = 1.5: the factor by which to increase the window size at each y level
# note that we should have larger windows for higher y (towards bottom)
# as the objects we detect (cars) will come closer to the cameraand appear larger in the image
# same car: if far away (small y) -> small window; if closer (larg y) -> large window
# we increase the window size linearly with y for simplicity
def slidingWindow(image_size, init_size=(64, 64), x_overlap=0.5, y_step=0.05,
                  x_range=(0, 1), y_range=(0, 1), scale=1.5):
    """
    Run a sliding window across an input image and return a list of the
    coordinates of each window.

    Window travels the width of the image (in the +x direction) at a range of
    heights (toward the bottom of the image in the +y direction). At each
    successive y, the size of the window is increased by a factor equal to
    @param scale. The horizontal search area is limited by @param x_range
    and the vertical search area by @param y_range.

    @param image_size (int, int): Size of the image (width, height) in pixels.
    @param init_size (int, int): Initial size of of the window (width, height)
        in pixels at the initial y, given by @param y_range[0].
    @param x_overlap (float): Overlap between adjacent windows at a given y
        as a float in the interval [0, 1), where 0 represents no overlap
        and 1 represents 100% overlap.
    @param y_step (float): Distance between successive heights y as a
        fraction between (0, 1) of the total height of the image.
    @param x_range (float, float): (min, max) bounds of the horizontal search
        area as a fraction of the total width of the image.
    @param y_range (float, float) (min, max) bounds of the vertical search
        area as a fraction of the total height of the image.
    @param scale (float): Factor by which to scale up window size at each y.
    @return windows: List of tuples, where each tuple represents the
        coordinates of a window in the following order: (upper left corner
        x coord, upper left corner y coord, lower right corner x coord,
        lower right corner y coord).
    """

    

    windows = []
    h, w = image_size[1], image_size[0]
    # 遍历垂直方向的 y 坐标（从 y_range[0]*h 到 y_range[1]*h），步长为 y_step*h。
    # 对于每个 y，窗口大小按距离初始 y 的偏移以 scale 因子线性增长（原始实现简化为：
    # win_size = init_size + scale * (y - y_start)）。注意这不是以倍数递增的精确金字塔，
    # 但能产生随高度变大的窗口集合以检测不同尺度的目标。
    for y in range(int(y_range[0] * h), int(y_range[1] * h), int(y_step * h)):
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * h))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        # 如果窗口超出垂直搜索区或宽度超过图像宽度，则停止生成
        if y + win_height > int(y_range[1] * h) or win_width > w:
            break
        # 横向步长根据重叠比例计算（重叠越大，步长越小）
        x_step = int((1 - x_overlap) * win_width)
        for x in range(int(x_range[0] * w), int(x_range[1] * w), x_step):
            windows.append((x, y, x + win_width, y + win_height))

    return windows


def display_windows(img: str, color=(0, 0, 255), thick=6):
    """
    Shows all windows of slidingWindow() in an image

    @param img:     path of img you want to show
    @param color:   windows' color drawn on img
    @param thick:   width of windows' edges
    @return         an img on which windows are drawn
    """
    image = plt.imread(img)
    h, w, c = image.shape
    windows = slidingWindow((w, h))
    rects = []
    for w in windows:
        rects.append(((int(w[0]), int(w[1])), (int(w[2]), int(w[3]))))

    random_color = False
    # Iterate through windows
    for rect in rects:
        if color == 'random' or random_color:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            random_color = True
        # Draw a rectangle given windows coordinates
        cv2.rectangle(image, rect[0], rect[1], color, thick)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

    return image

import cv2
import numpy as np
import sys
from image import Image

def side_by_side(before, after, normalize=True, resize=False):
    """
        Shows a side by side comparison of the before and after images.
        If the images are to be shown with cv2.imshow, normalize parameter
        must be set to True(because imshow requires array in [0, 1] interval.
        If the images are to be written to a file, normalize parameter must
        be set to False(because imwrite requires [0, 255] array interval.
    """

    horizontal_stack = np.hstack((before, after))
    if normalize:
        horizontal_stack /= 255

    if resize:
        cv2.namedWindow("Before & After", cv2.WINDOW_NORMAL)
        screen_size = (1024, 768)
        cv2.imshow("Before & After", cv2.resize(horizontal_stack, screen_size))
    else:
        cv2.imshow("Before & After", horizontal_stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
        Entry point of the application.
    """
    image_name = "off_white_balance.jpg"
    # image_name = "bear_noisy.jpg"
    # image_name = "overexposed.jpg"
    # image_name = "underexposed.jpg"
    # image_name = "lena_noisy.png"

    image = Image()
    try:
        image.open(image_name)
    except IOError as e:
        print(e)
        sys.exit(1)
    #
    before = image.img
    # before = image.grayscaled

    # after = image.correct_exposure()
    after = image.balance_white(cutoff=5)
    # after = image.denoise_channel(image.grayscaled, weight=25, error_tolerance=1e-8, iterations=500)
    side_by_side(before, after, normalize=False, resize=True)
    # cv2.imwrite("eu.jpg", after)
    # root = tkinter.Tk()
    # app = Application(master=root)
    # app.mainloop()


if __name__=="__main__":
    main()
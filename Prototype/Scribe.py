import cv2 as cv
import numpy as np
import os

DIR_PATH = "Source_images"
FINAL_DIR = "Finale_images"

class Scriber():
    KEY_RESET = ord("r")
    KEY_QUIT = ord("q")
    KEY_DELETE = ord("d")
    KEY_SAVE = ord("s")
    thicks = []
    coordinates = []
    thickness = 1
    click_count = 0
    win_name = "Test"
    img_index = 0
    img_name = ""
    img = ""

    def __init__(self, path2images, path2finale):
        self.final_path = path2finale
        self.path2dir = path2images
        self.next_img()
        self.void_img = self.create_void_img()

    def next_img(self):
        files = os.listdir(self.path2dir)
        if len(files) <= self.img_index:
            return False
        else:
            self.img_name = files[self.img_index]
            self.img = self.resize()
            self.img_index += 1
            return True

    def create_void_img(self):
        img = np.zeros((480, 720))
        return img

    def img_reset(self):
        self.img = self.resize()

    def resize(self):

        img = cv.imread(self.path2dir + "/" + self.img_name)
        final_wide = 720
        final_high = 480
        dim = (final_wide, final_high)
        resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
        return resized

    def __mouse_callback(self, event, x, y, flags, params):

        if event == cv.EVENT_LBUTTONDOWN:
            self.thicks.append(self.thickness)
            self.coordinates.append((x, y))
            self.click_count += 1

            if (self.click_count % 2) == 0:
                self.__handle_click_progress()

    def __handle_click_progress(self):
        length = len(self.coordinates)
        i = 0
        while i < length:
            cv.line(self.void_img, self.coordinates[i], self.coordinates[i+1], (255), self.thicks[i])
            cv.line(self.img, self.coordinates[i], self.coordinates[i+1], (0, 0, 255), self.thicks[i])
            i += 2

    def process(self):
        if __name__ == '__main__':
            def nothing(*arg):
                pass

        cv.namedWindow("settings")
        cv.createTrackbar('w', 'settings', 1, 255, nothing)

        while True:
            cv.namedWindow(self.win_name, cv.WINDOW_GUI_EXPANDED)
            cv.setMouseCallback(self.win_name, self.__mouse_callback)
            self.thickness = cv.getTrackbarPos('w', 'settings')
            cv.imshow("result", self.void_img)
            cv.imshow(self.win_name, self.img)
            quit = cv.waitKey(113)
            delete = cv.waitKey(100)
            save = cv.waitKey(115)
            reset = cv.waitKey(114)

            if quit == Scriber.KEY_QUIT:
                break

            if save == Scriber.KEY_SAVE:
                cv.imwrite(self.final_path + "/Source/" + self.img_name, self.resize())
                cv.imwrite(self.final_path + "/Marked/" + self.img_name, self.void_img)
                self.thicks.clear()
                self.coordinates.clear()
                self.void_img = self.create_void_img()
                if (self.next_img() == False):
                    break

            if delete == Scriber.KEY_DELETE:
                if len(self.coordinates) == 0:
                    continue
                self.coordinates.pop()
                self.coordinates.pop()
                self.thicks.pop()
                self.thicks.pop()
                self.img_reset()
                self.void_img = self.create_void_img()
                self.__handle_click_progress()

            if reset == Scriber.KEY_RESET:
                if len(self.coordinates) == 0:
                    continue
                self.thicks.clear()
                self.coordinates.clear()
                self.img_reset()
                self.void_img = self.create_void_img()
                self.__handle_click_progress()


scriber = Scriber(DIR_PATH, FINAL_DIR)
scriber.process()
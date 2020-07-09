import cv2 as cv

IMG_PATH = "Images/img1.jpg"
class Scriber():
    KEY_RESET = ord("r")
    KEY_QUIT = ord("q")
    KEY_DELETE = ord("d")

    def __init__(self, path2image):
        self.path = path2image
        self.thickness = 1
        self.img_name = "Test"
        self.img = self.resize()
        self.coordinates = []
        self.thicks = []
        self.click_count = 0

        cv.namedWindow(self.img_name, cv.WINDOW_GUI_EXPANDED)
        cv.setMouseCallback(self.img_name, self.__mouse_callback)

    def img_reset(self):
        self.img = self.resize()

    def resize(self):
 
        img = cv.imread(self.path)
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
        lenght = len(self.coordinates)
        print(lenght)
        print(self.coordinates)
        i = 0
        while i < lenght:
            cv.line(self.img, self.coordinates[i], self.coordinates[i+1], (0, 0, 255), self.thicks[i])
            i += 2

    def process(self):
        if __name__ == '__main__':
            def nothing(*arg):
                pass

        cv.namedWindow("settings")
        cv.createTrackbar('w', 'settings', 1, 255, nothing)

        while True:
            self.thickness = cv.getTrackbarPos('w', 'settings')
            cv.imshow(self.img_name, self.img)
            quit = cv.waitKey(113)
            delete = cv.waitKey(100)
            reset = cv.waitKey(114)

            if quit == Scriber.KEY_QUIT:
                break

            if delete == Scriber.KEY_DELETE:
                if len(self.coordinates) == 0:
                    continue
                self.coordinates.pop()
                self.coordinates.pop()
                self.thicks.pop()
                self.thicks.pop()
                self.img_reset()
                self.__handle_click_progress()

            if reset == Scriber.KEY_RESET:
                if len(self.coordinates) == 0:
                    continue
                self.thicks.clear()
                self.coordinates.clear()
                self.img_reset()
                self.__handle_click_progress()



scriber = Scriber(IMG_PATH)
scriber.process()
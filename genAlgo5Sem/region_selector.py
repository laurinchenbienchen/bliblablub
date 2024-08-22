import cv2

class RegionSelector:
    def __init__(self, image):
        self.image = image
        self.temp_image = image.copy()  # Kopie des Bildes für temporäre Zeichnungen
        self.regions = []
        self.drawing = False
        self.start_x, self.start_y = -1, -1

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Erstelle eine temporäre Kopie des Bildes für das Ziehen des Rechtecks
                self.temp_image = self.image.copy()
                end_x, end_y = x, y
                # Zeichne ein Rechteck in Pastellblau
                cv2.rectangle(self.temp_image, (self.start_x, self.start_y), (end_x, end_y), (173, 216, 230), 2)  # Pastellblau
                cv2.imshow("Image", self.temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            end_x, end_y = x, y
            self.drawing = False
            # Zeichne das endgültige Rechteck auf dem Originalbild
            cv2.rectangle(self.image, (self.start_x, self.start_y), (end_x, end_y), (0, 255, 255), 2)  # Cyan für das finale Rechteck
            self.regions.append((self.start_x, self.start_y, end_x - self.start_x, end_y - self.start_y))
            cv2.imshow("Image", self.image)

    def select_region(self):
        cv2.imshow("Image", self.image)
        cv2.setMouseCallback("Image", self.click_and_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.regions

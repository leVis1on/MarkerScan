import cv2
import sys
import numpy as np
from PIL import Image, ImageChops
import math
from PyQt6.QtWidgets import QApplication, QWidget, QFileDialog
from MarkerScanUI import Ui_MarkerScan

size = 8000

target_image = ''
trouble_text = ''
sample_images = ['samples/sample_110.png', 'samples/sample_110_alt.png']

class MarkerScan(QWidget):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MarkerScan()
        self.ui.setupUi(self) 

        self.ui.picButton.clicked.connect(self.openImageDialog)
        self.ui.runButton.clicked.connect(self.runProgram)

        self.show()

    def openImageDialog(self):
        global target_image
        file_name, _ = QFileDialog.getOpenFileName(self, 'Выберите полученное изображение', '', 'Изображения (*.png *.jpg *.bmp *.gif)')
        target_image = file_name
        if file_name != '':
            self.ui.picButton.setStyleSheet("QToolButton {\n"
            "background: rgba(0,0,0,0);\n"
            "border-radius: 40px;\n"
            "border: 8px solid #0F0;\n"
            "}\n"
            "QToolButton:hover {\n"
            "border-radius: 40px;\n"
            "border: 8px solid #000;\n"
            "}")
        else:
            self.ui.picButton.setStyleSheet("QToolButton {\n"
            "background: rgba(0,0,0,0);\n"
            "border-radius: 40px;\n"
            "border: 8px solid #F00;\n"
            "}\n"
            "QToolButton:hover {\n"
            "border-radius: 40px;\n"
            "border: 8px solid #000;\n"
            "}")

    def runProgram(self):
        global target_image
        global sample_images 
        global trouble_text
        trouble_text = ""

        self.ui.errorLabel.setText("")

        if target_image != "":
            if self.ui.circleMode.isChecked():
                sample_id = 0
                
                cropped_sample = ImageCrop(sample_images[sample_id])
                cropped_target = ImageCrop(target_image)
                
                points_sample = CircleDetection(cropped_sample)
                points_target = CircleDetection(cropped_target)
    
                if points_target is not None:

                    coords_sample = []
                    for point in points_sample[0]:
                        array = []
                        array.append(point[0])
                        array.append(point[1])
                        coords_sample.append(array)

                    coords_sample, _ = CircleSorting(coords_sample)
                    
                    coords_target = []
                    for point in points_target[0]:
                        array = []
                        array.append(point[0])
                        array.append(point[1])
                        coords_target.append(array)
                    
                    coords_target_attempt, max_deviation = CircleSorting(coords_target)
                    

                    white_image = np.ones((size, size), dtype=np.uint8) * 255

                    color_image = cv2.cvtColor(white_image, cv2.COLOR_GRAY2BGR)

                    CircleMarking(color_image, coords_target_attempt, (255, 0, 0))
                    
                    CircleMarking(color_image, coords_sample, (0, 0, 0))  

                    delta = DeltaCompute(coords_sample, coords_target_attempt)

                    condition = (np.abs(delta[:, 0]) > 20) | (np.abs(delta[:, 1]) > 20)

                    indices = np.where(condition)[0]

                    indexes = []
                    
                    for index in indices:
                        indexes.append(index + 1)

                    if indexes == []:
                        SaveFile(self, Image.fromarray(color_image))

                        Image.fromarray(color_image).show()

                        self.ui.errorLabel.setText("")
                        self.ui.errorLabel.setStyleSheet("color: rgb(0, 255, 0);")
                    else:
                        counter = max_deviation - 10

                        while indexes != []:
                            coords_target_attempt = sorted(coords_target, key=lambda x: (x[1] // counter, x[0], x[1], x[0]))  

                            delta = DeltaCompute(coords_sample, coords_target_attempt)

                            condition = (np.abs(delta[:, 0]) > 20) | (np.abs(delta[:, 1]) > 20)

                            indices = np.where(condition)[0]

                            indexes = []

                            for index in indices:
                                indexes.append(index + 1)

                            counter+=1

                        white_image = np.ones((size, size), dtype=np.uint8) * 255

                        color_image = cv2.cvtColor(white_image, cv2.COLOR_GRAY2BGR)

                        CircleMarking(color_image, coords_target_attempt, (255, 0, 0))
                
                        CircleMarking(color_image, coords_sample, (0, 0, 0))

                        SaveFile(self, Image.fromarray(color_image))

                        Image.fromarray(color_image).show()

                    self.ui.errorLabel.setText("")
                    self.ui.errorLabel.setStyleSheet("color: rgb(0, 255, 0);")

                else:
                    self.ui.errorLabel.setText(trouble_text)
                    self.ui.errorLabel.setStyleSheet("color: rgb(255, 0, 0);")
            elif self.ui.lineMode.isChecked():
                sample_id = 1

                imgCv_1 = LineImgCrop(sample_images[sample_id])
                imgCv_2 = LineImgCrop(target_image)

                if (imgCv_1 is not None) and (imgCv_2 is not None):
                
                    imgPIL_1 = Image.fromarray(imgCv_1)
                    imgPIL_2 = Image.fromarray(imgCv_2)

                    bitImg1 = imgPIL_1.convert('1')
                    bitImg2 = imgPIL_2.convert('1')

                    result = ImageChops.logical_xor(bitImg1, bitImg2)

                    new_image = Image.new("RGBA", result.size, (0, 0, 0, 0))

                    pixels = result.load()
                    new_pixels = new_image.load()

                    for i in range(result.size[0]):
                        for j in range(result.size[1]):

                            if pixels[i, j] == 0:
                                new_pixels[i, j] = (255, 255, 255, 255)

                            else:
                                new_pixels[i, j] = (255, 0, 0, 255)

                    original_image = Image.new("RGBA", bitImg1.size, (0, 0, 0, 0))

                    original_pixels = bitImg1.load()
                    new_original_pixels = original_image.load()

                    for i in range(bitImg1.size[0]):
                        for j in range(bitImg1.size[1]):

                            if original_pixels[i, j] == 0:
                                new_original_pixels[i, j] = (0, 0, 0, 255)

                            else:
                                new_original_pixels[i, j] = (0, 0, 0, 0)

                    result_image = Image.new('RGB', bitImg1.size)
                    result_image.paste(new_image, (0, 0))
                    result_image.paste(original_image, (0, 0), original_image)

                    SaveFile(self, result_image)

                    result_image.show()
            else:
                self.ui.errorLabel.setText("Выберите режим")
        else:
                self.ui.errorLabel.setText("Изображение не выбрано")

def LineImgCrop(img):
    global trouble_text

    image = cv2.imread(img)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 128

    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    inverted_image = cv2.bitwise_not(binary_image)

    marker_ids = [0, 1, 2, 3]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(inverted_image)

    if ids is not None:

        if len(ids) == 4:

            ordered_corners = [None] * 4

            for i in range(4):
                idx = np.where(ids == marker_ids[i])
                ordered_corners[i] = corners[idx[0][0]]

            left_top_corner = (ordered_corners[0][0][0][0], ordered_corners[0][0][0][1])
            right_top_corner = (ordered_corners[1][0][1][0], ordered_corners[1][0][1][1])
            right_bottom_corner = (ordered_corners[2][0][2][0], ordered_corners[2][0][2][1])
            left_bottom_corner = (ordered_corners[3][0][3][0], ordered_corners[3][0][3][1])

            source_points = np.array([left_top_corner, right_top_corner, right_bottom_corner, left_bottom_corner], dtype=np.float32)

            width, height = 4600, 4600

            destination_points = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(source_points, destination_points)
            result_image = cv2.warpPerspective(inverted_image, matrix, (width, height))


            return result_image
            
        else:
            trouble_text = "Количество маркеров не равно четырем"
    else:
        trouble_text = "Не найдены маркеры"

def MarkerDetection(img):
    global trouble_text

    marker_ids = [0, 1]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(img)

    if ids is not None:
        
        if len(ids) == 2:
            
            ordered_corners = [None] * 2

            for i in range(2):
                idx = np.where(ids == marker_ids[i])
                ordered_corners[i] = corners[idx[0][0]]
                
            return ordered_corners
        else:
            trouble_text = f'Количество маркеров не равно двум, оно равно {len(ids)}'
    else:
        trouble_text = 'Не найдены маркеры'

def CircleDetection(img):
    global trouble_text

    binary_blurred = cv2.GaussianBlur(img, (9, 9), 0)

    circles = cv2.HoughCircles(
        binary_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=13,
        minRadius=7,
        maxRadius=20
    )

    if circles is not None:

        circles = np.int16(np.around(circles))
            
        return circles

    else:
        trouble_text = "Круги не обнаружены на изображении."

def CircleSorting(circles):
    circles = sorted(circles, key=lambda x: (x[1] // 131, x[0], x[1], x[0]))

    max_deviation = abs(circles[0][0] - circles[1][0])

    circles = sorted(circles, key=lambda x: (x[1] // max_deviation, x[0], x[1], x[0]))

    return circles, max_deviation

def ImageCrop(img):
    global radius

    global trouble_text

    image = cv2.imread(img)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 110

    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    inverted_image = cv2.bitwise_not(binary_image)

    ordered_corners = MarkerDetection(inverted_image)

    if ordered_corners is not None:
            
        point1 = (int(ordered_corners[0][0][0][0]), int(ordered_corners[0][0][0][1]))
        point2 = (int(ordered_corners[1][0][1][0]), int(ordered_corners[1][0][1][1]))
        
        vector1 = (point2[0] - point1[0], point2[1] - point1[1])
        vector2 = (point2[0] - point1[0], point1[1] - point1[1])

        length1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        length2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        angle_radians = math.acos(dot_product / (length1 * length2))

        angle_degrees = math.degrees(angle_radians)
        
        if point2[1] < point1 [1]:
            rotation_matrix = cv2.getRotationMatrix2D((inverted_image.shape[1] / 2, inverted_image.shape[0] / 2), 360-angle_degrees, 1)
        else:
            rotation_matrix = cv2.getRotationMatrix2D((inverted_image.shape[1] / 2, inverted_image.shape[0] / 2), angle_degrees, 1)

        aligned_image = cv2.warpAffine(inverted_image, rotation_matrix, (inverted_image.shape[1], inverted_image.shape[0]))

        ordered_corners = MarkerDetection(aligned_image)

        if ordered_corners is not None:

            point1 = (ordered_corners[0][0][0][0] - 500, ordered_corners[0][0][0][1] - 100)
            point2 = (ordered_corners[1][0][1][0] + 500, ordered_corners[1][0][1][1] - 100)

            x1, y1 = point1  
            x2, _ = point2  

            width = x2 - x1
            height = width - 400

            point1 = (ordered_corners[0][0][0][0], ordered_corners[0][0][0][1])
            point2 = (ordered_corners[0][0][2][0], ordered_corners[0][0][2][1])
            point3 = (ordered_corners[1][0][1][0], ordered_corners[1][0][1][1])
            point4 = (ordered_corners[1][0][0][0], ordered_corners[1][0][0][1])
            point5 = (ordered_corners[1][0][2][0], ordered_corners[1][0][2][1])

            cv2.rectangle(aligned_image, (int(point1[0] - 30), int(point1[1]) - 30), (int(point2[0]) + 30, int(point2[1]) + 30), 255, -1)
            cv2.rectangle(aligned_image, (int(point4[0]- 30), int(point4[1]) - 30), (int(point5[0] + 30), int(point5[1]) + 30), 255, -1)
            cv2.circle(aligned_image, (int(point1[0]) + 10, int(point1[1]) + 10), 10, 0, -1)
            cv2.circle(aligned_image, (int(point3[0]) - 10, int(point3[1]) + 10), 10, 0, -1)

            cropped_square = aligned_image[int(y1):int(y1+height), int(x1):int(x1+width)] 

            circles = CircleDetection(cropped_square)

            if circles is not None:

                circles, _ = CircleSorting(circles[0])

                if  len(circles) == 361:

                    center_x, center_y, radius = circles[180]

                    image_pillow = Image.fromarray(cropped_square)

                    width, height = image_pillow.size

                    new_width, new_height = size, size
                    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

                    x_offset = (new_width // 2) - center_x
                    y_offset = (new_height // 2) - center_y

                    new_image.paste(image_pillow, (x_offset, y_offset))

                    result_image = np.array(new_image)

                    gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

                    threshold_value = 1

                    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

                    return binary_image

                else:
                    trouble_text = f'Количество кругов не равно 359. Оно равно {len(circles)}'
            else:
                trouble_text = "Круги не обнаружены на изображении."  

def CircleMarking(img, circles, color):
    text = 1

    for circle in circles:
        x, y = circle

        cv2.circle(img, (x, y), 6, color, -1)
        cv2.putText(img, str(text), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        text+=1

def DeltaCompute(sample, target):
    k = 0.0212765957446

    delta = np.array(sample, dtype=np.float64) - np.array(target, dtype=np.float64)
    
    for point in delta:
        x, y = point
        if y == 0.0:
            y = y * k
        else:
            y = y * k * -1
        x = x * k
        point[0] = round(x, 3)
        point[1] = round(y, 3)

    return delta

def SaveFile(self, img):
    file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить файл", "", "Изображение (*.png)")
    if file_name:
        img.save(file_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_window = MarkerScan()
    sys.exit(app.exec())
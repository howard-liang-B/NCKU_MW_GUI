'''
。button 類
  * btn_open_file: 開啟檔案用按鈕
  * btn_zoom_in: 放大用按鈕
  * btn_zoom_out: 縮小用按鈕
。label 類
  * label_ratio: 顯示現在圖片縮放比例
  * label_file_name: 顯示現在檔名
  * label_img_shape: 顯示現在/原來圖片大小
  * label_img: 顯示圖片
。silder 類
  * silder_zoom: 控制圖片大小
  * scrollArea 類
  * scrollArea: 圖片縮放區域

----------------------------------------------

。 set_instance_name：使用者可以 call，去修改一些想要的變化，並在此實作複雜功能與算法。
。 __update_instance_name：private function，不希望使用者去 call，主要只負責單純的更新 info，
   而不實作任何複雜功能或算法。

----------------------------------------------

init()：初始化
read_file_and_init()：實作讀取圖片檔案，並實作圖片初始化
set_img_ratio()：設定圖片比率，並實作變化
set_path()：更換檔案時，更新圖片路徑
set_zoom_in()：設定 zoom in 功能
set_zoom_out()：設定 zoom out 功能
set_slider_value()：設定縮放功能的那條 bar
__update_img()：更新圖片
__update_text_file_path()：更新圖片路徑的文字
__update_text_ratio()：更新圖片縮放比率的文字
__update_text_img_shape()：更新圖片大小的文字
'''


from PyQt5 import QtCore 
from PyQt5.QtGui import QImage, QPixmap
import cv2

## 顯示原圖的類別
class img_controller(object):
    def __init__(self, img_path, label_img, label_file_path, label_ratio):
        # (str)
        self.img_path = img_path

        # 以下都是 Ui 的 label 物件
        self.label_img = label_img # 顯示照片的 
        self.label_file_path = label_file_path # 顯示照片路徑的
        self.label_ratio = label_ratio # 顯示縮放比例的
        self.ratio_value = 50
        self.read_file_and_init()
        self.__update_img()

    def read_file_and_init(self):
        try:
            self.img = cv2.imread(self.img_path)
            self.origin_height, self.origin_width, self.origin_channel = self.img.shape 
            bytesPerline = 3 * self.origin_width
            self.qimg = QImage(self.img, self.origin_width, self.origin_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.origin_qpixmap = QPixmap.fromImage(self.qimg)
            self.ratio_value = 50        
            self.set_img_ratio()           
        except Exception as e:
            print(f"Error reading file and initializing: {e}")
            self.origin_height, self.origin_width, self.origin_channel = 0, 0, 0   

    # 設定圖片比率，並實作變化
    def set_img_ratio(self):
        try:
            self.ratio_rate = pow(10, (self.ratio_value - 50)/50)
            qpixmap_height = self.origin_height * self.ratio_rate
            self.qpixmap = self.origin_qpixmap.scaledToHeight(int(qpixmap_height))
            self.__update_img() # 顯示更新後的圖片
            self.__update_text_file_path() # 顯示影像路徑
            self.__update_text_ratio() # 顯示縮放倍率
        except Exception as e:
            print(f"Error set_img_ratio(img_controller): {e}")

    def set_path(self, img_path):
        self.img_path = img_path
        self.read_file_and_init()
        self.__update_img()

    # 更新圖片時，同步增加監聽偵測滑鼠位置的 mousePressEvent
    def __update_img(self):       
        try:
            self.label_img.setPixmap(self.qpixmap)
            self.label_img.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            '''
            這行程式碼將 self.label_img.mousePressEvent 設置為 self.get_clicked_position 方法。
            這意味著當 label_img 接收到滑鼠按下事件時，將調用 get_clicked_position 方法來處理這個事件。
            '''
        except Exception as e:
            print(f"Error __update_img(img_controller): {e}")

    # 更新圖片路徑的文字
    def __update_text_file_path(self):
        self.label_file_path.setText(f"File path = {self.img_path}")

    # 更新圖片圖片縮放比率的文字
    def __update_text_ratio(self):
        self.label_ratio.setText(f"{int(100*self.ratio_rate)} %")

    # 讀取縮放功能的那條 bar
    def set_slider_value(self, value):
        self.ratio_value = value
        self.set_img_ratio()

    ##############################################################################################
    # 顯示切割照片的 bbox，並顯示文字在 bbox 上方
    ##############################################################################################
    def show_tooth_bbox(self, bbox_list, idx):
        self.__draw_rectangle(bbox_list, idx)
    
    def __draw_rectangle(self, bbox_list, idx):
        img_to_draw_tooth = self.img.copy() # 因為是拿來畫bbox的圖，所以不能留下之前的紀錄，所以另外拿一個變數
        center_x, center_y, w, h = bbox_list[idx]

        # 計算矩形的左上角和右下角的座標
        top_left_x = int(center_x - w / 2)
        top_left_y = int(center_y - h / 2)
        bottom_right_x = int(center_x + w / 2)
        bottom_right_y = int(center_y + h / 2)

        # 畫矩形 (BGR 顏色, 例如藍色)
        text_color = (222, 222, 222)
        background_color = (200, 10, 10)  # 藍色
        thickness = 3  # 線條粗細

        # 獲取文字尺寸
        cur_text = f'tooth {idx+1}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(cur_text, font, font_scale, thickness)

        cv2.rectangle(img_to_draw_tooth, (top_left_x, top_left_y - text_height - 15), 
                      (top_left_x + text_width, top_left_y + baseline - 10), background_color, -1)
        cv2.putText(img_to_draw_tooth, cur_text, (top_left_x, top_left_y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        cv2.rectangle(img_to_draw_tooth, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), background_color, thickness)

        # 換成 GUI(PyQt) 顯示格式
        self.origin_height, self.origin_width, self.origin_channel = img_to_draw_tooth.shape 
        bytesPerline = 3 * self.origin_width
        self.qimg = QImage(img_to_draw_tooth, self.origin_width, self.origin_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.origin_qpixmap = QPixmap.fromImage(self.qimg)       
        self.set_img_ratio() 

## 顯示切割牙齒的類別
class crop_img_controller(object):
    def __init__(self, img, label_img, label_ratio):
        self.img = img
        self.label_img = label_img
        self.label_ratio = label_ratio
        self.ratio_value = 50
        self.read_file_and_init()

    # TODO 
    def read_file_and_init(self):
        try:
            # 注意 self.img 必須是 3D
            self.origin_height, self.origin_width, self.origin_channel = self.img.shape
            bytesPerline = 3 * self.origin_width

            # 將 numpy 陣列轉換為 bytes 對象
            self.qimg = QImage(self.img.tobytes(), self.origin_width, self.origin_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.origin_qpixmap = QPixmap.fromImage(self.qimg)
            self.ratio_value = 50
            self.set_img_ratio() 
        except Exception as e:
            print(f"Error reading file and initializing(crop_img_controller): {e}")
            self.origin_height, self.origin_width, self.origin_channel = 0, 0, 0

    # 設定圖片比率，並實作變化
    def set_img_ratio(self):
        try:
            self.ratio_rate = pow(10, (self.ratio_value - 50)/50) 
            qpixmap_height = self.origin_height * self.ratio_rate
            self.qpixmap = self.origin_qpixmap.scaledToHeight(int(qpixmap_height))
            self.__update_img() # 顯示更新後的圖片
            self.__update_text_ratio() # 顯示縮放倍率
        except Exception as e:
            print(f"Error set_img_ratio(crop_img_controller): {e}")

    # 更新"圖片"
    def __update_img(self):  
        try:     
            self.label_img.setPixmap(self.qpixmap)
            self.label_img.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            '''
            這行程式碼將 self.label_img.mousePressEvent 設置為 self.get_clicked_position 方法。
            這意味著當 label_img 接收到滑鼠按下事件時，將調用 get_clicked_position 方法來處理這個事件。
            '''
        except Exception as e:
            print(f"Error __update_img(crop_img_controller): {e}")

    # 更新圖片縮放比率的"文字"
    def __update_text_ratio(self):
        self.label_ratio.setText(f"{int(100*self.ratio_rate)} %")

    # 讀取縮放功能的那條 bar(獲得滑動 bar 時的數值)
    def set_slider_value(self, value):
        self.ratio_value = value
        self.set_img_ratio()

    # 傳入要顯示的切割的牙齒
    def set_crop_tooth(self, crop_tooth_img):
        self.img = crop_tooth_img
        self.read_file_and_init()
    
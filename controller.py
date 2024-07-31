
# GUI 套件
from PyQt5 import QtCore 
from PyQt5.QtWidgets import QMainWindow, QFileDialog

# 其他套件
import time
import os
import cv2
import pandas

# 我的套件
from medical_ui import Ui_Dentist_app
from img_controller import img_controller, crop_img_controller
from yolov8_predictor import yolo_seg
from mask_rcnn_predictor import Mask_RCNN_predictor, mask_processor
from data_writer import excel_writer

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_Dentist_app()
        self.ui.setupUi(self)
        self.setup_control()

    ##############################################################################################
    # 設置好照片控制物件 & 按鈕連結
    ##############################################################################################
    def setup_control(self):
        self.filepath = "" # image 路徑
        self.excel_path = "" # excel 路徑
        self.cur_file_dir = "./" # 選擇的照片的資料夾
        self.cur_excel_dir = "./" # 選擇的excel的資料夾

        # 顯示原圖照片的物件
        self.original_img_controller = img_controller(img_path="",
                                             label_img=self.ui.label_original_img,
                                             label_file_path=self.ui.label_img_path,
                                             label_ratio=self.ui.label_ratio_1)
        
        # 顯示切割照片的物件
        self.tooth_img_controller = crop_img_controller(img=None,
                                             label_img=self.ui.label_crop_tooth,
                                             label_ratio=self.ui.label_ratio_2)
        
        # 先建立 yolov8 物件，然後傳入路徑去分割圖片，(Mask-RCNN 同理)
        self.set_yolov8_seg_obj()
        self.set_mask_rcnn_obj()

        # 建立寫入或寫出的excel的物件
        self.set_excel_obj()
        
        # pushButton
        self.ui.btn_select_img.clicked.connect(self.open_file) # 選擇照片的按鈕
        self.ui.pushButton_import.clicked.connect(self.import_file)
        self.ui.pushButton_export.clicked.connect(self.export_file)
        self.ui.pushButton_save.clicked.connect(self.save_note) # 儲存寫入的資訊

        self.ui.comboBox_select_tooth.currentIndexChanged.connect(self.show_crop_tooth)

        self.ui.slider_zoom_1.valueChanged.connect(self.getslidervalue) # 控制原圖大小的滑桿
        self.ui.slider_zoom_2.valueChanged.connect(self.getslidervalue) # 控制切割出來的單顆牙齒的滑桿



    ##############################################################################################
    # 顯示照片的函式
    ##############################################################################################
    # 選擇要打開的照片
    def open_file(self):
        try:
            self.ui.pushButton_save.setText(f'save *')
            self.ui.label_status.setText(f'Status: \nLoading image ...')
            self.filepath, filetype = QFileDialog.getOpenFileName(self, "Open file", self.cur_file_dir) # start path
            self.cur_file_dir = os.path.dirname(self.filepath) 

            self.init_new_picture()
            self.init_patient_data() # 選擇新照片時，name、phone、note 都會清空
            self.ui.progressBar_status.setValue(20)
            print("== initial picture & patient data.")

            self.imgpath_to_yolov8_seg()
            self.ui.progressBar_status.setValue(40)
            print("== imgpath_to_yolov8_seg.")
            
            self.get_masks_list() # TODO 
            self.ui.progressBar_status.setValue(70) 
            print("== get_masks_list.")

            self.process_masks()
            self.ui.progressBar_status.setValue(90)
            print("== process_masks.")

            self.show_crop_tooth()
            self.ui.progressBar_status.setValue(100)
            print("== show_crop_tooth.")
            
            self.ui.label_status.setText(f'Status: \nSuccess loading image.')
        except Exception as e:
            print("(func) open_file(): ", e)
            self.ui.label_status.setText(f'Status: \n No image selected or predict failed.')

    # 初始化照片大小，並顯示在GUI
    def init_new_picture(self):
        self.ui.slider_zoom_1.setProperty("value", 50)
        self.ui.slider_zoom_2.setProperty("value", 50) 
        self.original_img_controller.set_path(self.filepath)

    def init_patient_data(self):
        self.ui.textEdit_name.setText("")
        self.ui.textEdit_phone.setText("")
        self.ui.textEdit_note.setText("")    

    # 滑桿被移動時，讀取滑桿的值
    def getslidervalue(self):
        self.button = self.sender()
        if self.button == self.ui.slider_zoom_1:
            self.original_img_controller.set_slider_value(self.ui.slider_zoom_1.value()+1) # 原圖的滑桿
        else:
            self.tooth_img_controller.set_slider_value(self.ui.slider_zoom_2.value()+1) # 單顆牙齒的滑桿

    # 顯示選擇的 crop tooth (! 同時呼叫前面的 show_tooth_bbox!)
    def show_crop_tooth(self):
        # 獲取 tooth n 的 n
        tooth_idx = self.ui.comboBox_select_tooth.currentIndex() + 1

        # 傳入 numpy(但要轉換成 bytes) 照片給函式
        try:
            self.tooth_img_controller.set_crop_tooth(self.tooth_list[tooth_idx - 1]) # 因為傳入的是 1、2、3 ... # TODO 
            self.original_img_controller.show_tooth_bbox(self.tooth_bboxes_xywh, tooth_idx - 1)
        except Exception as e:
            print(f"Error select tooth: {e}")    

    ##############################################################################################
    # YOLOv8 切割牙齒 
    ##############################################################################################
    # YOLO 切割的物件
    def set_yolov8_seg_obj(self):
        self.img_yolo_predictor = yolo_seg()

    def imgpath_to_yolov8_seg(self):
        self.img_yolo_predictor.read_img(self.filepath)

        # 回傳切割的照片和座標，儲存在 tooth_list
        self.tooth_list, self.tooth_bboxes_xywh = self.img_yolo_predictor.get_crop_tooth() 
        self.__set_comboBox_count(len(self.tooth_list))

    # 將偵測的數量加入下拉式選單(comboBox)
    def __set_comboBox_count(self, num_of_crop_tooth):
        comboBox_len = self.ui.comboBox_select_tooth.count()
        while num_of_crop_tooth != comboBox_len:
            if num_of_crop_tooth < comboBox_len:
                self.ui.comboBox_select_tooth.removeItem(comboBox_len - 1)
            elif num_of_crop_tooth > comboBox_len:
                self.ui.comboBox_select_tooth.addItem(f'Tooth {comboBox_len + 1}')
            comboBox_len = self.ui.comboBox_select_tooth.count()
        print(f"(func) __set_comboBox_count, comboBox {len(self.ui.comboBox_select_tooth)} boxes.")


    ##############################################################################################
    # Mask R-CNN 
    ##############################################################################################
    # 這邊建立 Mask-RCNN 物件，同時建立處理 mask 的物件
    def set_mask_rcnn_obj(self):
        self.Mask_RCNN_obj = Mask_RCNN_predictor()
        self.Mask_processor = mask_processor()

    def get_masks_list(self): # TODO  太久了
        self.Mask_RCNN_obj.load_crop_tooth_list(self.tooth_list)
        self.tooth_masks_list, self.crown_masks_list, self.bone_masks_list = self.Mask_RCNN_obj.get_masks() 
        print(f'(func) get_masks_list, get_masks.')

    def process_masks(self):
        self.Mask_processor.load_masks_list(self.tooth_masks_list, self.crown_masks_list, self.bone_masks_list)
        cej_points_list, alc_points_list = self.Mask_processor.get_points_list()
        print(f'(func) process_masks, get_points_list.')
        self.__draw_points(cej_points_list, alc_points_list)

    def __draw_points(self, cej_points_list, alc_points_list):
        for i in range(len(self.tooth_list)):
            tooth = cv2.cvtColor(self.tooth_list[i], cv2.COLOR_GRAY2BGR)
            tooth = cv2.circle(tooth, cej_points_list[i][0], 10, [0, 255, 0], 2)
            tooth = cv2.circle(tooth, cej_points_list[i][1], 10, [0, 255, 0], 2)
            tooth = cv2.circle(tooth, alc_points_list[i][0], 10, [0, 0, 255], 2)
            tooth = cv2.circle(tooth, alc_points_list[i][1], 10, [0, 0, 255], 2)
            self.tooth_list[i] = tooth
        print(f'(func) __draw_points, draw circle.')

    ##############################################################################################
    # 將 "Patient Name"、"Phone Number"、"Note" 寫入 excel 儲存起來 
    ##############################################################################################
    def set_excel_obj(self):
        self.excel_writer_obj = excel_writer() # 建立 excel_writer 物件

    def import_file(self):
        self.excel_path, filetype = QFileDialog.getOpenFileName(self, "Import Excel", self.cur_excel_dir)
        self.cur_excel_dir = os.path.dirname(self.excel_path) 
        self.ui.textEdit_excel_name.setText(os.path.basename(self.excel_path))
        self.excel_writer_obj.read_excel(self.excel_path)
        if self.excel_path != "":
            self.ui.label_status.setText(f'Status: \nImport "{os.path.basename(self.excel_path)}".')
        else:
            self.ui.label_status.setText(f'Status: \nImport failed.')
 
    # 匯出 excel 檔案 !
    def export_file(self):
        try:
            self.excel_dir = QFileDialog.getExistingDirectory(self, "Export Excel", self.cur_excel_dir)
            save_path = os.path.join(self.excel_dir, self.ui.textEdit_excel_name.toPlainText())

            # 如果是匯入的excel那就會有".xlsx"則無需再加入，但沒有匯入舊excel則須加入".xlsx"
            if ".xlsx" not in save_path:
                save_path += ".xlsx"

            self.excel_writer_obj.save_excel(save_path) # 儲存資料
            self.ui.label_status.setText(f'Status: \nExport "{self.ui.textEdit_excel_name.toPlainText()}".')
        except Exception as e:
            self.ui.label_status.setText(f"Status: \nPlease enter excel name.")
            print(f'export_file: {e}')

    def save_note(self):
        self.excel_writer_obj.save_data(self.ui.textEdit_name, 
                                                  self.ui.textEdit_phone,
                                                  self.filepath,
                                                  self.ui.textEdit_note)
        self.ui.label_status.setText(f'Status: \nSave {self.ui.textEdit_name.toPlainText()} data.')
        self.ui.pushButton_save.setText(f'save')
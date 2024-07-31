import cv2
from ultralytics import YOLO

class yolo_seg(object):
    def __init__(self):
        self.__init_lists()

    def __init_lists(self):
        self.crop_tooth_list = [] # 儲存切割的照片
        self.xywh_pts_list = [] # 儲存切割的 bbox 座標

    def read_img(self, img_path):
        self.img_path = img_path
        self.img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        self.__CLAHE_img()
    
    def __CLAHE_img(self):
        filtered_image = cv2.medianBlur(self.img, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.CLAHE_img = clahe.apply(filtered_image)

    def get_crop_tooth(self):
        self.__init_lists()
        self.__predict() # 預測
        self.__get_bboxes_xywh() # 獲取預測的 bbox 的座標
        self.__get_bboxes_conf() # 獲取預測的 bbox 信心值
        self.__crop_img() # 利用得到的 bbox 座標切割照片
        return self.crop_tooth_list, self.xywh_pts_list

    def __predict(self):
        model = YOLO('YOLOv8_model/yolov8_tooth_segment.pt')
        self.results = model(self.img_path) # 不能直接傳入 numpy 給他預測

    def __get_bboxes_xywh(self):
        r = self.results[0] # 因為每次都只讀取一張照片，所只需要 [0] 的部分
        self.xywh_pts_list = r.boxes.xywh.tolist() # 切割照片的座標
        ''' self.xywh_pts_list 長這樣
        [[230.85365295410156, 1061.594970703125, 461.7073059082031, 1122.968505859375], 
        [499.7344970703125, 870.0240478515625, 350.6777648925781, 1188.279052734375],
        [1263.2779541015625, 1066.2049560546875, 114.53271484375, 1156.91259765625], ......] 
        '''

    def __get_bboxes_conf(self):
        r = self.results[0]
        self.conf_list = r.boxes.conf.tolist()
        ''' self.conf_list 長這樣
        [0.9597697257995605, 0.9553078413009644, 0.9541887044906616, 0.9442502856254578, 0.7620221972465515, ......] 
        '''

    def __crop_img(self):
        for i in range(len(self.xywh_pts_list)):
            # 信心值為小於 0.6 不切割
            if self.conf_list[i] < 0.6:
                continue
            center_x, center_y, width, height = self.xywh_pts_list[i]
            center_x, center_y, width, height = int(center_x), int(center_y), int(width), int(height)
            half_width, half_height = width//2, height//2

            crop_tooth = self.CLAHE_img[center_y-half_height:center_y+half_height, center_x-half_width:center_x+half_width]
            self.crop_tooth_list.append(crop_tooth) # 將切割的照片儲存在一個 list 裡面
        print(f"## YOLOv8, __crop_img, crop {len(self.xywh_pts_list)} tooth.")


class yolo_classify(object):
    def __init__(self, tooth_list):
        self.tooth_list = tooth_list # 1.
        self.tooth_class_list = [] # 2.
        
    def get_crop_tooth_class(self):
        self.__classify()
        return self.tooth_class_list

    def __classify(self):
        model = YOLO('model/yolov8_tooth_classify.pt')

        for crop_tooth in self.tooth_list:
            self.results = model(crop_tooth) # 3.
            self.__get_top1_class()

    def __get_top1_class(self):
        r = self.results[0]
        class_list = r.names # 儲存類別的字典 --> {0: 'defective', 1: 'good'}
        probs_obj = r.probs
        top1_class_name = class_list[probs_obj.top1] # probs_obj.top1 類別的 idx
        self.tooth_class_list.append(top1_class_name) # 儲存此單顆牙齒是 defective 還是 good

if __name__ == "__main__":
    ## YOLOv8 segment
    # yolo_predictor_1 = yolo_seg("PA7.jpg")
    # yolo_predictor_1.get_crop_tooth()

    ## YOLOv8 classify
    img = cv2.imread(r"C:\Users\user\Desktop\IoT Medical\GUI\PA7.jpg")
    tooth_list = [img[10:200, :], img[30:300, :], img[40:500]]
    # yolo_classifier = yolo_classify(tooth_list)
    # tooth_class_list = yolo_classifier.get_crop_tooth_class()
    # print(tooth_class_list)
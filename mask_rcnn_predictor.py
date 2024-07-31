# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np # version  numpy==2.0
import os, json, cv2, random
import torch

# import some common detectron2 utilities
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo # version  pillow==8.4.0
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


############################################################################################################
############################################################################################################

# This object uses Mask-RCNN to obtain masks for three different types.
class Mask_RCNN_predictor(object):
   def __init__(self):
      self.__init_lists()
      self.__register_dataset()
      self.__create_predictor()

   def __init_lists(self):
      # 繪製出遮罩
      self.tooth_masks_list = []
      self.crown_masks_list = []
      self.bone_masks_list = []

   def load_crop_tooth_list(self, crop_tooth_list):
      self.crop_tooth_list = crop_tooth_list

   def get_masks(self): # TODO  跑太久
      print("## Mask RCNN, start get_masks ok.")
      self.__init_lists()
      self.__start_predict()
      print("## Mask RCNN, __start_predict ok.")
      return self.tooth_masks_list, self.crown_masks_list, self.bone_masks_list

   def __register_dataset(self):
      # 這裡建立的 dataset 是為了讓類別名稱是 (tooth、crown、bone)，所以只建立一個(有可能會出錯)
      register_coco_instances("my_dataset_train", {}, "datasets/tooth_train_2.json", r"C:\Users\user\Desktop\Detectron2\datasets\2_train")
      self.train_metadata = MetadataCatalog.get("my_dataset_train")
      self.train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

   def __create_predictor(self):
      self.cfg_tooth = get_cfg()
      self.cfg_tooth.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
      self.cfg_tooth.DATASETS.TRAIN = ("my_dataset_train",)
      self.cfg_tooth.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
      self.cfg_tooth.MODEL.WEIGHTS = os.path.join("Mask_RCNN_models", "tooth_model/model_final.pth")  # Use the trained model
      self.cfg_tooth.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Set to the number of your custom classes
      self.tooth_predictor = DefaultPredictor(self.cfg_tooth)

      self.cfg_crown = get_cfg()
      self.cfg_crown.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
      self.cfg_crown.DATASETS.TRAIN = ("my_dataset_train",)
      self.cfg_crown.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
      self.cfg_crown.MODEL.WEIGHTS = os.path.join("Mask_RCNN_models", "crown_model/model_final.pth")  # Use the trained model
      self.cfg_crown.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Set to the number of your custom classes
      self.crown_predictor = DefaultPredictor(self.cfg_crown)

      self.cfg_bone = get_cfg()
      self.cfg_bone.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
      self.cfg_bone.DATASETS.TRAIN = ("my_dataset_train",)
      self.cfg_bone.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
      self.cfg_bone.MODEL.WEIGHTS = os.path.join("Mask_RCNN_models", "bone_model/model_final.pth")  # Use the trained model
      self.cfg_bone.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Set to the number of your custom classes
      self.bone_predictor = DefaultPredictor(self.cfg_bone)

   def __start_predict(self):
      for i in range(len(self.crop_tooth_list)):
         # load image
         crop_tooth_img = self.crop_tooth_list[i] 
         crop_tooth_img = cv2.cvtColor(crop_tooth_img, cv2.COLOR_GRAY2RGB)
         print("## Mask RCNN, crop_tooth_img cvt ok.")

         # preidct 
         tooth_outputs = self.tooth_predictor(crop_tooth_img) 
         crown_outputs = self.crown_predictor(crop_tooth_img) # TODO  有時候crown masks 會少預測
         bone_outputs = self.bone_predictor(crop_tooth_img)
         print("## Mask RCNN, predictor ok.")

         # draw predict masks
         self.__draw_masks(tooth_outputs, "tooth")
         self.__draw_masks(crown_outputs, "crown")
         self.__draw_masks(bone_outputs, "bone")
         print("## Mask RCNN, draw points ok.")

   # TODO 
   def __draw_masks(self, outputs, output_class):
      print("1")
      try:
         class_masks = {
               output_class: torch.zeros_like(outputs["instances"].pred_masks[0], dtype=torch.uint8, device=torch.device("cuda:0"))
         }
      except Exception as e:
         print(f"## Mask RCNN, (func) __draw_masks: {e}")

      print("2")
      largest_masks = {}
      # Assign a unique integer label to each object in the mask
      for i, pred_class in enumerate(outputs["instances"].pred_classes):
         class_name = self.train_metadata.thing_classes[pred_class]
         if output_class not in class_name:
            continue

         current_mask = outputs["instances"].pred_masks[i].to(device=torch.device("cuda:0"))
         current_area = current_mask.sum().item()
         
         # If we haven't seen this class yet or the current mask is larger, update the largest mask
         if class_name not in largest_masks or current_area > largest_masks[class_name][1]:
               largest_masks[class_name] = (current_mask, current_area, i + 50)

      # Create masks only for the largest objects of each class
      for class_name, (largest_mask, _, unique_label) in largest_masks.items():
         class_masks[class_name] = torch.where(largest_mask, unique_label, class_masks[class_name])

      print("3")
      # Save the masks for each class with unique integer labels
      for class_name, class_mask in class_masks.items():
         # Convert the tensor to a NumPy array and then to a regular (CPU) array
         class_mask_np = class_mask.cpu().numpy()

         # Save the image with unique integer labels
         if output_class == "tooth":
            self.tooth_masks_list.append(class_mask_np.astype(np.uint8))
         elif output_class == "crown":
            self.crown_masks_list.append(class_mask_np.astype(np.uint8))
         elif output_class == "bone":
            self.bone_masks_list.append(class_mask_np.astype(np.uint8))

############################################################################################################
############################################################################################################

# This class is use to process masks to get "CEJ"、"ALC"
class mask_processor(object):
   def __init__(self):
      self.__init_points_lists()

   def __init_points_lists(self):
      self.cej_points_list = []
      self.alc_points_list = []

   def load_masks_list(self, tooth_masks_list, crown_masks_list, bone_masks_list):
      self.tooth_masks_list = tooth_masks_list
      self.crown_masks_list = crown_masks_list
      self.bone_masks_list = bone_masks_list

   def get_points_list(self):
      # 將儲存 CEJ、ALC 的 list 清空
      self.__init_points_lists()
      print(f'@@ len of tooth_masks_list: {len(self.tooth_masks_list)}')
      for mask_idx in range(len(self.tooth_masks_list)):
         print(f'mask_idx: {mask_idx}, len(tooth): {len(self.tooth_masks_list)}, len(crown): {len(self.crown_masks_list)} \
               , len(bone): {len(self.bone_masks_list)}')

         # get current masks, and convert to 3 dimension
         cur_tooth = cv2.cvtColor(self.tooth_masks_list[mask_idx], cv2.COLOR_GRAY2RGB)
         cur_crown = cv2.cvtColor(self.crown_masks_list[mask_idx], cv2.COLOR_GRAY2RGB)
         cur_bone = cv2.cvtColor(self.bone_masks_list[mask_idx], cv2.COLOR_GRAY2BGR)

         self.__get_cej_points(cur_tooth, cur_crown)
         print("@@ cej_points ok.")
         self.__get_alc_points(cur_tooth, cur_bone) # TODO  很久
         print("@@ alc_points ok.")

      return self.cej_points_list, self.alc_points_list

      
   def __get_cej_points(self, cur_tooth, cur_crown):
      # 存放 cej level 的每個點，為了找出最左右兩邊的點
      cur_cej_level_points_list = []
      h, w, _ = cur_tooth.shape

      # 對crown做膨脹
      kernel = np.ones((9, 9), np.uint8) # change 
      crown_dilated = cv2.dilate(cur_crown, kernel, iterations=3) # change 
      print("@@ 1")

      # 調整 crown_dilated 的大小以匹配 cur_tooth(有時候怪怪的，會不一樣大小，所以才要調整成一樣大小)
      if crown_dilated.shape != cur_tooth.shape:
         crown_dilated = cv2.resize(crown_dilated, (w, h))

      # compute value
      tooth_value = np.max(cur_tooth)
      crown_value = np.max(cur_crown)
      new_tooth_value = crown_value * 2
      res_max_value = new_tooth_value + crown_value

      # update tooth value
      self.__set_new_tooth_mask_value(cur_tooth, tooth_value, new_tooth_value)
      print("@@ 2")

      # 疊加 # TODO 
      print(f'@@ {cur_tooth.shape}, {crown_dilated.shape}')
      cej_level_img = np.clip(cur_tooth + crown_dilated, 0, 255) # change 
      print("@@ 3")
      

      # find CEJ level
      for i in range(1, w-1):
         for j in range(1, h-1):
               pixel_value_list = (cej_level_img[j-1:j+1, i-1:i+1]).flatten()
               if new_tooth_value in pixel_value_list and res_max_value in pixel_value_list:
                  cur_cej_level_points_list.append([j, i])
      
      # 排列
      cur_cej_level_points_list = sorted(cur_cej_level_points_list, key=lambda x: x[1]) # 按照 i 排列(x軸順序排列)

      # 找出最左邊跟最右邊 [ [[cej_left_y, cej_left_x], [cej_right_y, cej_right_x]], [[], []] ......]
      self.cej_points_list.append([cur_cej_level_points_list[0][::-1], cur_cej_level_points_list[-1][::-1]])
      print("@@ 4")

   
   def __get_alc_points(self, cur_tooth, cur_bone):
      # 存放 alc level 的每個點，為了找出最左右兩邊的點
      cur_alc_level_points_list = []
      h, w, _ = cur_tooth.shape

      # find gap
      tooth_value = np.max(cur_tooth)
      bone_value = np.max(cur_bone)
      new_tooth_value = bone_value * 2

      # update tooth value 
      self.__set_new_tooth_mask_value(cur_tooth, tooth_value, new_tooth_value)

      # compute pixel value in kernel
      kernel_size = 5
      for i in range(w - kernel_size):
         for j in range(h - kernel_size):
               value_list = np.concatenate((
                cur_tooth[j:j+kernel_size, i-kernel_size//2:i+kernel_size//2].flatten(),
                cur_bone[j:j+kernel_size, i-kernel_size//2:i+kernel_size//2].flatten()
               ))

               if (new_tooth_value in value_list) and (bone_value in value_list) and (0 in value_list) \
                   and np.array_equal(cur_bone[j, i], [0, 0, 0]):
                  cur_alc_level_points_list.append([j, i])

      # 排列
      cur_alc_level_points_list = sorted(cur_alc_level_points_list, key=lambda x: x[1]) # 按照 i 排列(x軸順序排列)

      # 找出最左邊跟最右邊 [ [[alc_left_y, alc_left_x], [alc_right_y, alc_right_x]], [[], []] ......]
      self.alc_points_list.append([cur_alc_level_points_list[0][::-1], cur_alc_level_points_list[-1][::-1]])


   # 設置新的值給 tooth mask
   def __set_new_tooth_mask_value(self, cur_tooth, old_tooth_value, new_tooth_value):
      cur_tooth[cur_tooth == old_tooth_value] = new_tooth_value
      #################################################################
      # 下面這段太慢
      # h, w, _ = cur_tooth.shape
      # for i in range(w):
      #    for j in range(h):
      #          if np.max(cur_tooth[j, i]) == old_tooth_value:
      #             cur_tooth[j, i] = np.full(3, new_tooth_value)
      #################################################################

    

if __name__ == "__main__":
   ########################################################################################################################
   # Mask_RCNN_predictor
   crop_tooth_list = []
   crop_tooth_list.append(cv2.imread("PA teeth\\11_.jpg"))
   crop_tooth_list.append(cv2.imread("PA teeth\\46_2.jpg"))
   crop_tooth_list.append(cv2.imread("PA teeth\\63_3.jpg"))

   Mask_rcnn_obj = Mask_RCNN_predictor()
   Mask_rcnn_obj.load_crop_tooth_list(crop_tooth_list)
   tooth_masks_list, crown_masks_list, bone_masks_list = Mask_rcnn_obj.get_masks()

   # mask_processor
   Mask_processor = mask_processor()
   Mask_processor.load_masks_list(tooth_masks_list, crown_masks_list, bone_masks_list)
   cej_points_list, alc_points_list = Mask_processor.get_points_list()
   for i in range(len(crop_tooth_list)):
      tooth = crop_tooth_list[i]
      tooth = cv2.circle(tooth, cej_points_list[i][0], 10, [0, 255, 0], 2)
      tooth = cv2.circle(tooth, cej_points_list[i][1], 10, [0, 255, 0], 2)
      tooth = cv2.circle(tooth, alc_points_list[i][0], 10, [255, 0, 0], 2)
      tooth = cv2.circle(tooth, alc_points_list[i][1], 10, [255, 0, 0], 2)
      cv2.imshow("tooth", tooth)
      cv2.waitKey()
      cv2.destroyAllWindows()


   ########################################################################################################################
   # # Read Image
   # image = cv2.imread("PA teeth/46_2.jpg") # BGR

   # # Register datasets(為了讓預測的類別名稱是我們命名的)
   # register_coco_instances("my_dataset_train", {}, "datasets/bone_train_2.json", r"C:\Users\user\Desktop\Detectron2\datasets\2_train")
   # train_metadata = MetadataCatalog.get("my_dataset_train")
   # train_dataset_dicts = DatasetCatalog.get("my_dataset_train")


   # # Configuration for inference with instance segmentation
   # cfg_inst = get_cfg()
   # cfg_inst.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
   # cfg_inst.DATASETS.TRAIN = ("my_dataset_train",)
   # cfg_inst.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
   # cfg_inst.MODEL.WEIGHTS = os.path.join("models", "bone_model/model_final.pth")  # Use the trained model
   # cfg_inst.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Set to the number of your custom classes


   # # Inference with the custom trained model
   # cfg_inst.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set a custom testing threshold
   # predictor = DefaultPredictor(cfg_inst)
   # outputs = predictor(image)


   # # 將預測結果獲取出來
   # '''
   # 1. image[:, :, ::-1]將圖像從BGR格式轉換為RGB格式 , 因為OpenCV讀取的圖像默認是BGR格式 , 而Visualizer期望的是RGB格式。
   # 2. MetadataCatalog.get(cfg_inst.DATASETS.TRAIN[0])獲取數據集的元數據(metadata) , 包括類別名稱、顏色等。
   #    cfg_inst.DATASETS.TRAIN[0]是配置中訓練數據集的名稱。
   # 3. outputs["instances"]包含了模型的實例分割預測結果，通常包括邊界框、類別標籤和掩碼等信息。
   #    outputs["instances"].to("cpu")將預測結果從GPU移動到CPU , 因為Visualizer需要在CPU上進行處理。
   # '''
   # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg_inst.DATASETS.TRAIN[0]), scale=1) # RGB
   # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

   # cv2.imshow("image", out.get_image()[:, :, ::-1])
   # cv2.waitKey()
   # cv2.destroyAllWindows()
   ########################################################################################################################
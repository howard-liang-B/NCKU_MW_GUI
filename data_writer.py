import pandas as pd
import os
from datetime import datetime

class excel_writer(object):
    def __init__(self):
        self.excel_path = None
        self.excel_data = pd.DataFrame(columns=["Date", # 年/月/日 - 小時:分鐘:秒數
                                                "Patient Name", 
                                                "Phone Number", 
                                                "Image ID", 
                                                "Note"])

    def read_excel(self, excel_path):
        self.excel_path = excel_path
        try:
            self.old_excel_data = pd.read_excel(self.excel_path) # 讀取之前已儲存的 excel
            self.excel_data = pd.concat([self.old_excel_data, self.excel_data], ignore_index=True)
        except Exception as e:
            print("read_excel(no dir select): ", e)

    def save_data(self, patient_name, phone_number, image_path, note):
        cur_t = datetime.now()
        new_excel_data = pd.DataFrame({'Date': [f'{cur_t.year}/{cur_t.month}/{cur_t.day} - {cur_t.hour}:{cur_t.minute}:{cur_t.second}'],
                                       'Patient Name': [patient_name.toPlainText()], 
                                       'Phone Number': [phone_number.toPlainText()], 
                                       'Image ID': [os.path.basename(image_path)],
                                       'Note': [note.toPlainText()]})
        self.excel_data = pd.concat([self.excel_data, new_excel_data], ignore_index=True)

    def save_excel(self, save_path):
        print("~~~~~~ start save excel ~~~~~")
        self.excel_data.to_excel(save_path, index=False)
        print("~~~~~~ end save excel ~~~~~")



if __name__ == "__main__":
    results = pd.DataFrame(columns=["Patient Name", "Phone Number", "Note"])
    print(results)
    print("------------")

    new_data = pd.DataFrame({'Patient Name': ["Howard"], 'Phone Number': ["0966520060"], 'Note': ['Not cure']})
    results = pd.concat([results, new_data], ignore_index=True)
    new_data = pd.DataFrame({'Patient Name': ["Daniel"], 'Phone Number': ["0909879787"], 'Note': ['Haha cure']})
    results = pd.concat([results, new_data], ignore_index=True)
    print(results)
    print("------------")
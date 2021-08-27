from glob import glob
from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import makeImageFolder
import digit_recog_resnet152_val as  digit_val

class DigitGaugeDataset(Dataset):
    def __init__(self, file_json, transform):
        list_image, list_value, dict_json_info = makeImageFolder.get_Image_Value_List_from_json(file_json)

        self.list_image = list_image
        self.list_label = list_value
        self.dict_json_info = dict_json_info
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.list_image)

    def __getitem__(self, idx):
        image = self.list_image[idx].convert('RGB')
        image = self.transform(image)
        return image, self.list_label[idx]


def getValueFromJson(file_json, model_path_load):
    args = digit_val.make_args()
    args.checkpoint_format = model_path_load
    model = digit_val.load_model(args)


    gaugedataset = DigitGaugeDataset(file_json, digit_val.resnet_transform )
    len_digit = len(gaugedataset)

    gauge_data_loader = torch.utils.data.DataLoader(gaugedataset, batch_size=len_digit, shuffle=False)
    data, target = next(iter(gauge_data_loader))

    target = target.numpy()
    model.eval()
    with torch.no_grad():
        if args.cuda:
            data = data.cuda()
        output = model(data)
        output = output.cpu().numpy().argmax(axis=1)

    list_match = target ==  output
    accuracy = list_match.tolist().count(True) / len(list_match)
    print(f'target:{target}, output:{output} ==> accuracy:{accuracy} : file_json:{file_json}')


if __name__ == '__main__':

    dir_digit_test = r'D:\proj_gauge\민성기\digitGaugeTest'
    list_json = glob(dir_digit_test + r"\*.json")
    print(f'len(list_json) is {len(list_json)}')

    model_filename = 'resnet152-100.pt'

    for file_json in list_json :
        npred = getValueFromJson(file_json, model_path_load=model_filename)
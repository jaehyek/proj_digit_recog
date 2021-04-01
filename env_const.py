dict_cmd_makeImageFolder = { 'cmd':'makeImageFolder', 'folder_json':r'D:\proj_gauge\민성기\digitGaugeSamples' , 'folder_digit':r'.\digit_class'}
dict_cmd_imageAugmentation = { 'cmd':'imageAugmentation', 'dir_in':r'.\digit_class' , 'dir_out':r'.\digit_class_aug'}
dict_cmd_makeTrainValidFromDigitClass = { 'cmd':'makeTrainValidFromDigitClass', 'dir_digit_class':r'.\digit_class_aug' , 'dir_train':r'.\digit_class_train', 'dir_valid':r'.\digit_class_valid', 'train_ratio':0.8}

dict_cmd_DigitRecogModel = { 'cmd':'DigitRecogModel', 'dir_train':r'.\digit_class_aug' , 'dropout': 0.2, 'phase':'training', 'model_path_load':r'./model.pt','model_path_save':r'./model.pt' }
dict_cmd_getValueFromJson = { 'cmd':'getValueFromJson', 'file_json':r'D:\proj_gauge\민성기\digitGaugeSamples\10061-56490.json' , 'model_path_load':r'./model.pt' }

dict_cmd_serverclientTest = { 'cmd':'serverclientTest', 'param0': 'hello world',  'param1': 12345.567 }
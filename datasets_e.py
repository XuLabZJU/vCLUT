import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class ImageDataset_test(Dataset):
    def __init__(self, root, mode="test", selfpaird_data_type="WBC"):
        self.mode = mode
        self.selfpaird_data = selfpaird_data_type
        file = open(os.path.join(root, self.selfpaird_data + '.txt'),'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        for i in range(len(test_input_files)):             
            self.test_input_files.append(os.path.join(root,test_input_files[i][:-1] + ".bmp"))

    def __getitem__(self, index):
        if self.mode == "test":    
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
        img_input = TF.to_tensor(img_input)
        return {"A_input": img_input, "input_name": img_name}

    def __len__(self):
        if self.mode == "test":
            return len(self.test_input_files)
        
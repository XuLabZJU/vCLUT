import argparse
import time
import torch
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models_e import *
from datasets_e import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="weights", help="directory of saved models")
parser.add_argument("--data_type", type=str, default="WBC", help="load the weights model")

opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
classifier = Classifier_selfpaired()   
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    classifier = classifier.cuda()

LUTs = torch.load("%s/VCLUTs_%s.pth" % (opt.model_dir, opt.data_type))
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
LUT0.eval()
LUT1.eval()
LUT2.eval()
classifier.load_state_dict(torch.load("%s/classifier_%s.pth" % (opt.model_dir, opt.data_type)))
classifier.eval()

dataloader = DataLoader(
    ImageDataset_test("./data/%s" % opt.data_type,  mode="test", selfpaird_data_type = opt.data_type),
    batch_size=1,
    shuffle=False,
    #num_workers=1,  # Linux
    num_workers=0,   # win
)

def generator(img):
    pred = classifier(img).squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT 
    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT,img)
    return combine_A

def visualize_result():
    out_dir1 = "results/%s/output" % opt.data_type
    os.makedirs(out_dir1, exist_ok=True)
    out_dir2 = "results/%s/output_identified" % opt.data_type
    os.makedirs(out_dir2, exist_ok=True)
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        img_name = batch["input_name"]
        #t
        fake_B = generator(real_A)
        save_image(fake_B, os.path.join(out_dir1,"%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)                             
        img_identified = torch.cat((real_A.data, fake_B.data), -1)
        save_image(img_identified, os.path.join(out_dir2,"%s.jpg" % (img_name[0][:-4])), nrow=1, normalize=False)

visualize_result() 

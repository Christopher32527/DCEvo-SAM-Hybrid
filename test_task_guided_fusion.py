import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size, check_requirements,
                           check_yaml, coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode
from models.yolo import Model

from sleepnet import (DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor,
                        CrossAttentionBlock,prominent)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
from torchvision import transforms as T


def test(pth_path='', out_path=''):
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.CRITICAL)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ckpt_path=pth_path
    ckpt_path= "ckpt/DCEvo_fusion_branch.pth"
    for dataset_name in ["M3FD", "FMB", "TNO","RoadScene"]:
        print("The test result of "+dataset_name+' :')
        test_folder=os.path.join('datasets/',dataset_name) 
        test_out_folder=os.path.join('datasets/', dataset_name, 'DCEvo_Fusion')
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Encoder = nn.DataParallel(DE_Encoder()).to(device)
        Decoder = nn.DataParallel(DE_Decoder()).to(device)
        LFExtractor = nn.DataParallel(LowFreqExtractor(dim=64)).to(device)
        HFExtractor = nn.DataParallel(HighFreqExtractor(num_layers=3)).to(device)
        CA1 = nn.DataParallel(CrossAttentionBlock()).to(device)
        CA3 = nn.DataParallel(CrossAttentionBlock()).to(device)
        CA5 = nn.DataParallel(CrossAttentionBlock()).to(device)
    
        model = DetectMultiBackend("ckpt/DCEvo_detect_branch.pt").to(device)   
    
        Encoder.load_state_dict(torch.load(ckpt_path)['DE_Encoder'])
        Decoder.load_state_dict(torch.load(ckpt_path)['DE_Decoder'])
        LFExtractor.load_state_dict(torch.load(ckpt_path)['LowFreqExtractor'])
        HFExtractor.load_state_dict(torch.load(ckpt_path)['HighFreqExtractor'])
        CA1.load_state_dict(torch.load(ckpt_path)['CA1'])
        CA3.load_state_dict(torch.load(ckpt_path)['CA3'])
        CA5.load_state_dict(torch.load(ckpt_path)['CA5'])
        
        Encoder.eval()
        Decoder.eval()
        LFExtractor.eval()
        HFExtractor.eval()   
        CA1.eval()  
        CA3.eval()  
        CA5.eval()  
        transform = T.Compose([T.Resize((768, 1024)), T.ToTensor()])
    
        with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder,"ir")):
    
                data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
    
                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
                
                
                _, _, h, w = data_VIS.shape
                reh = round(h/32.) * 32
                rew = round(w/32.) * 32
                data_VIS = nn.functional.interpolate(data_VIS, (reh, rew), mode='bilinear')
                data_IR = nn.functional.interpolate(data_IR, (reh, rew), mode='bilinear')
                
                ####################################################################################################################
                data_VIS_Det = image_read_cv2(os.path.join(test_folder,"images",img_name), mode='RGB')/255.0
                data_VIS_Det = torch.FloatTensor(data_VIS_Det)
                data_VIS_Det = data_VIS_Det.cuda()
                padding = (0, 0, 128, 128)  # (left, right, top, bottom)
                
                f2 = nn.functional.interpolate(data_VIS_Det.unsqueeze(0).permute(0, 3, 1, 2), (reh, rew), mode='bilinear')
                
                for ijk in range(5):
                    f2 = model.model.model[ijk].forward(f2)
                f3 = (model.model.model[6].forward(model.model.model[5].forward(f2)))
                fsppf = model.model.model[9].forward(model.model.model[8].forward(model.model.model[7].forward(f3)))
                
                f5 = model.model.model[12].forward(model.model.model[11].forward((model.model.model[10].forward(fsppf), f3)))
                f6 = model.model.model[15].forward(model.model.model[14].forward((model.model.model[13].forward(f5), f2)))

                feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
                    
                feature_V_B = CA1(feature_V_B, f6) + feature_V_B
                feature_V_D = CA1(feature_V_D, f6) + feature_V_D
                
                feature_I_B = CA3(feature_I_B, f6) + feature_I_B
                feature_I_D = CA3(feature_I_D, f6) + feature_I_D
                
                feature_V = CA5(feature_V, f6) + feature_V
                feature_I = CA5(feature_I, f6) + feature_I
                
                feature_F_B = LFExtractor(feature_I_B+feature_V_B)
                feature_F_D = HFExtractor(feature_I_D+feature_V_D)
                
                data_Fuse, feature_F = Decoder(data_VIS*0.5+data_IR*0.5, feature_F_B, feature_F_D) 
                
                ####################################################################################################################
                # DE-Decoder
                data_Fuse = prominent()(data_Fuse)
                ####################################################################################################################
                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
                
                data_Fuse = nn.functional.interpolate(data_Fuse, (h, w), mode='bilinear')
                
                fi = np.squeeze((data_Fuse * 255).cpu().numpy())
                
                img_save(fi, img_name.split(sep='.')[0], test_out_folder)
        print(str(len(os.listdir(os.path.join(test_folder,"ir")))) + " results have been saved in "+test_out_folder+'.')
            
if __name__ == '__main__':
    test()
        
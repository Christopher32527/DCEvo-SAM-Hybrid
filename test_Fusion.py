import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sleepnet import (DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor,)
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
    # ckpt_path=pth_path
    ckpt_path= "ckpt/DCEvo_fusion.pth"
    for dataset_name in ["M3FD", "FMB", "TNO","RoadScene"]:
        print("The test result of "+dataset_name+' :')
        test_folder=os.path.join('datasets/',dataset_name) 
        # test_out_folder=os.path.join('datasets/', dataset_name, 'DCEvo_Fusion_Only')
        test_out_folder=os.path.join('datasets/', dataset_name, 'images')
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Encoder = nn.DataParallel(DE_Encoder()).to(device)
        Decoder = nn.DataParallel(DE_Decoder()).to(device)
        LFExtractor = nn.DataParallel(LowFreqExtractor(dim=64)).to(device)
        HFExtractor = nn.DataParallel(HighFreqExtractor(num_layers=3)).to(device)
    
        # 加载模型权重,支持CPU和GPU
        checkpoint = torch.load(ckpt_path, map_location=device)
        Encoder.load_state_dict(checkpoint['DE_Encoder'])
        Decoder.load_state_dict(checkpoint['DE_Decoder'])
        LFExtractor.load_state_dict(checkpoint['LowFreqExtractor'])
        HFExtractor.load_state_dict(checkpoint['HighFreqExtractor'])
        
        Encoder.eval()
        Decoder.eval()
        LFExtractor.eval()
        HFExtractor.eval()   
        transform = T.Compose([T.Resize((768, 1024)), T.ToTensor()])
    
        with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder,"ir")):
    
                data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
    
                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
                
                feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
                
                feature_F_B = LFExtractor(feature_I_B+feature_V_B)
                feature_F_D = HFExtractor(feature_I_D+feature_V_D)
                
                data_Fuse, feature_F = Decoder(data_VIS*0.5+data_IR*0.5, feature_F_B, feature_F_D) 
                
                fi = np.squeeze((data_Fuse * 255).cpu().numpy())
                
                img_save(fi, img_name.split(sep='.')[0], test_out_folder)
        print(str(len(os.listdir(os.path.join(test_folder,"ir")))) + " results have been saved in "+test_out_folder+'.')

        
if __name__ == '__main__':
    test()
        
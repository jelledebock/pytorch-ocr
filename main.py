"""  
This script brings together the following pytorch text extraction and 
recognition demonstrators.

https://github.com/clovaai/CRAFT-pytorch
https://github.com/clovaai/deep-text-recognition-benchmark

All required files and libraries are documented in requirements.txt and
Python libs are in the modules and lib folders.
"""
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

import cv2
from skimage import io
from nltk.metrics.distance import edit_distance
from PIL import Image
import numpy as np
import json
import zipfile
from collections import OrderedDict

from lib import craft_utils
from lib import imgproc
from lib import file_utils
from lib.craft import CRAFT

from lib.utils import CTCLabelConverter, AttnLabelConverter, Averager
from lib.dataset import hierarchical_dataset, AlignCollate
from lib.ocr_model import Model

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

class PytorchTextExtractor:
    def __init__(self, extraction_model_path, ocr_model_path, use_gpu=True, write_debug=True):
        # Text extraction config
        self.extraction_model_path = extraction_model_path
        self.text_threshold = 0.7 #text confidence threshold
        self.low_text = 0.4 #text low-bound score
        self.link_threshold = 0.4 #link confidence threshold
        self.cuda = use_gpu #use GPU for inference
        self.refine = True #enable link refiner
        self.canvas_size = 1280
        self.mag_ratio = 1.5 # Magnification ratio
        self.refiner_model = './models/craft_refiner_CTW1500.pth'
        self.poly = False

        # General config
        self.result_folder = './output'
        self.write_debug = write_debug

        # Text recognition config
        self.recognition_workers = 4
        self.recognition_batch_size=192
        self.recognition_saved_model='./models/TPS-ResNet-BiLSTM-Attn.pth'
        self.recognition_batch_max_length=25
        self.recognition_imgH=32
        self.recognition_imgW=100
        self.recognition_character='0123456789abcdefghijklmnopqrstuvwxyz'
        self.recognition_num_fiducial = 20
        self.recognition_input_channel =1
        self.recognition_output_channel=512
        self.recognition_hidden_size=256

        self.recognition_Transformation='TPS'
        self.recognition_FeatureExtraction='ResNet'
        self.recognition_SequenceModeling='BiLSTM'
        self.recognition_Prediction = 'Attn'

        self.init_text_recognition()
        self.init_text_extraction()

    def init_text_extraction(self):
        # Initialize CRAFT network (text extraction)
        self.craft_net = CRAFT()
        sys.stdout.write("Loading text extraction model.\n")
        sys.stdout.flush()
        if self.cuda:
            self.craft_net.load_state_dict(copyStateDict(torch.load(self.extraction_model_path)))
        else:
            self.craft_net.load_state_dict(copyStateDict(torch.load(self.extraction_model_path, map_location='cpu')))

        if self.cuda:
            self.craft_net = self.craft_net.cuda()
            self.craft_net = torch.nn.DataParallel(self.craft_net)
            cudnn.benchmark = False

        self.craft_net.eval()  

        # LinkRefiner
        self.refine_net = None
        if self.refine:
            from lib.refinenet import RefineNet
            self.refine_net = RefineNet()
            sys.stdout.write('Loading weights of refiner from checkpoint (' + self.refiner_model + ')\n')
            sys.stdout.flush()
            if self.cuda:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.refiner_model, map_location='cpu')))

            self.refine_net.eval()
            self.poly = True 

        sys.stdout.write("Loading extraction model DONE.\n")
        sys.stdout.flush()

    def init_text_recognition(self):
        cudnn.benchmark = True
        cudnn.deterministic = True
        if self.cuda:
            self.recognition_device = torch.device('cuda')
            self.recognition_ngpu = torch.cuda.device_count()
        else:
            self.recognition_device = torch.device('cpu')

        self.recognition_converter = AttnLabelConverter(self.recognition_character)
        self.recognition_num_class = len(self.recognition_converter.character)
        self.recognition_input_channel = 1

        self.recognition_model = Model(self)
        self.recognition_model = torch.nn.DataParallel(self.recognition_model).to(self.recognition_device)
        
        sys.stdout.write("Loading pretrained recognition model {}.\n".format(self.recognition_saved_model))
        sys.stdout.flush()
        self.recognition_model.load_state_dict(torch.load(self.recognition_saved_model, map_location=self.recognition_device))
        self.recognition_exp_name =   '_'.join(self.recognition_saved_model.split(os.path.sep)[1:])

        if self.write_debug:
            os.makedirs(f'./result/{self.recognition_exp_name}', exist_ok=True)
            os.system(f'cp {self.recognition_saved_model} ./result/{self.recognition_exp_name}/')

        self.recognition_criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(self.recognition_device)  # ignore [GO] token = ignore index 0
        sys.stdout.write("Loading pretrained recognition model DONE. \n")
        sys.stdout.flush()

    def recognize_text(self, image):
        self.recognition_model.eval()
        image_tensor = imgproc.prepare_for_ocr(Image.fromarray(image).convert('L'))
        
        image_tensor = image_tensor.to(self.recognition_device)
        batch_size = image_tensor.size(0)

        length_for_pred = torch.IntTensor([self.recognition_batch_max_length] * batch_size).to(self.recognition_device)
        text_for_pred = torch.LongTensor(batch_size, self.recognition_batch_max_length + 1).fill_(0).to(self.recognition_device)

        preds = self.recognition_model(image_tensor, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = self.recognition_converter.decode(preds_index, length_for_pred)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        output = []
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in self.recognition_Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            output.append((pred, float(confidence_score)))

        return output[0]


    def extract_text(self, image, filename='test_image'):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.craft_net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if self.write_debug: 
            # save score text
            mask_file = os.path.join(self.result_folder, "{}.jpg".format(filename))
            cv2.imwrite(mask_file, ret_score_text)

            file_utils.saveResult(filename, image[:,:,::-1], polys, dirname=self.result_folder+os.path.sep)
            sys.stdout.flush()

        return boxes, polys, ret_score_text

    def get_cropped_texts_images(self, image, boxes, polys, ret_score_text, filename='output_cropped'):
        cropped_imgs = []
        for i, poly in enumerate(polys):
            left, right, top, bottom = imgproc.poly_to_lrtb(poly)

            cropped_img = image[int(top):int(bottom), int(left):int(right)]
            cropped_imgs.append(cropped_img)

            if self.write_debug:
                text_output_file = os.path.join(self.result_folder, "res_{}_bbox_{}.jpg".format(filename, i))
                cv2.imwrite(text_output_file, cropped_img)

        return cropped_imgs

    def process_frame(self, image, fn=None):
        if not fn:
            fn = 'frame'

        # Text location extraction
        boxes, polys, ret_score_text = self.extract_text(image, filename=fn)
        images_cropped = self.get_cropped_texts_images(image, boxes, polys, ret_score_text, filename=fn)
 
        predictions_text = []
        predictions=[]
        # Recognition
        for i, img in enumerate(images_cropped):
            print(img.shape)
            if img.shape[0]>0 and img.shape[1]>0:
                text, certainty = self.recognize_text(img)
            else:
                text, certainty = '', 1
            
            left, right, top, bottom = imgproc.poly_to_lrtb(polys[i])
            bbox = np.array([[left, top],[right, top], [right, bottom], [left, bottom]])

            predictions.append((bbox, text, certainty))
            if self.write_debug:
                predictions_text.append("{} ({}%)".format(text, round(float(certainty*100))))

        if self.write_debug:
            file_utils.saveResult(fn+"annotated.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes, texts=predictions_text)
        
        return predictions

    def process_file(self, img_path):
        image = imgproc.loadImage(img_path)
        fn = img_path.split(os.path.sep)[-1].split('.')[0]

        return self.process_frame(image, fn=fn)


if __name__ == "__main__":
    text_extractor = PytorchTextExtractor('./models/craft_mlt_25k.pth',
        './models/TPS-ResNet-BiLSTM-Attn.pth', use_gpu=False,
        write_debug=True
    )
    print(text_extractor.process_file(sys.argv[1]))

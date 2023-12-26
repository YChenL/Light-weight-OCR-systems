from imghdr import tests
from operator import mod
import os
import time
import string
import argparse
import re

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from utils import CTCLabelConverter
import dataset
from model import Model
import itertools
from PIL import Image
import math
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NEG_INF = float('-inf')


import kenlm
# # kenlm_model = kenlm.Model('zhiyu_corpus_sep1.bin')
kenlm_model = kenlm.Model("/data/zhiyu/zhlmmodel_3.bin")
beam_size=3
search_depth=3
# parameters of 5-gram LM, zhiyuLM
# lm_panelty=1.1263157894736844
# len_bonus=2.9473684210526314

# parameters of 5-gram LM, generalLM
# lm_panelty=0.9210526315789473
# len_bonus=2.1578947368421053

# parameters of 3-gram LM, generalLM
lm_panelty=0.9210526315789473
len_bonus=2.1578947368421053
# 1.1263157894736844, 2.9473684210526314

char_set = []

class Beam(object):
    def __init__(self, prefix='',pb=float(0),pnb=NEG_INF):
        self.prefix = prefix
        self.pb = pb # probability of ending with blank
        self.pnb = pnb # probability of ending with non-blank
        self.plm = 0 # probability under LM

    def prob(self):
        return np.logaddexp(self.pb, self.pnb)

    def total(self):
        return np.logaddexp(self.pb,self.pnb) + self.plm



def str_normal(conts):
    '''
    消除中英文格式问题
    :param conts:
    :return:
    '''
    conts = conts.replace(' ', '').replace('	', '').replace('﹐',',').replace("；", ";")
    conts = conts.replace('（', '(').replace('）', ')').replace('|','')
    conts = conts.replace('：', ':').replace('一', '-').replace('—', '-')
    conts = conts.replace('，', ',').replace('“', '"').replace('”', '"')
    conts = conts.replace('【', '[').replace('】', ']').replace('0', 'O').replace("﻿","")

    return conts

def use_language(top_res):

    t_conts=re.sub("[$¥,.\- ]",'',top_res)
    if t_conts.isdigit() :
        return False
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    match = zhPattern.search(t_conts)
    if not match and not t_conts.isalpha():
        return False
    return True

def ctc_bm_decode(preds):

    print("ctc lm")

    preds =preds.log_softmax(2).permute(1, 0, 2)

    result_text = []
    pred_size, batch_size, _ = preds.shape

    preds = preds.cpu().numpy()
    topk_idx = np.flip(np.argsort(preds, axis=2), axis=2)[:, :, :search_depth]
    # print(topk_idx)
    greedy_result = []
    for b in range(batch_size):
        top_line = []
        top1_idx = topk_idx[:, b, 0]
        last_timestep = 0
        pred_size_len=0
        greedy_result_single=""
        for t in range(pred_size):
            if (top1_idx[t] != 0) and (top1_idx[t] != len(char_set) - 1) and (not (t > 0 and top1_idx[t - 1] == top1_idx[t])):
                top_line.append((char_set[top1_idx[t]], t))
                last_timestep = t
            pred_size_len=pred_size_len+1
            # if self.without_lm:
        for item in top_line:
            chr,i = item
            greedy_result_single+=chr
                # return greedy_result
        greedy_result.append(greedy_result_single)
        if not use_language(greedy_result_single):
            result_text.append(greedy_result_single)
            continue
        start_timestep=max(top_line[0][1]-2,0) if len(top_line)>0 else 0
        saved_beams = [Beam()]
        for t in range(pred_size):
            if t< start_timestep:
                continue
            if t>last_timestep+2:
                break
            suffix = ''.join(c[0] for c in itertools.dropwhile(lambda a: a[1] <= t, top_line))
            saved_beams = context_beam_search(saved_beams, visual_candidates=topk_idx[t, b, :],
                                                  preds_at_t=preds[t, b, :], suffix=suffix)

        result_text.append(saved_beams[0].prefix)

    return greedy_result,result_text

def context_beam_search(input_beams, visual_candidates, preds_at_t, suffix):
    combined_condidates = []
    combined_condidates = itertools.repeat(visual_candidates, len(input_beams))
    gen_beams = {}

    for input_beam, candidates in zip(input_beams, combined_condidates):
        for idx in candidates:
            prefix = input_beam.prefix
            p = preds_at_t[idx]
            if prefix not in gen_beams:
                gen_beams[prefix] = Beam(prefix=prefix, pb=NEG_INF, pnb=NEG_INF)
            if idx == 0:  # 碰到了blank
                gen_beams[prefix].pb = np.logaddexp(gen_beams[prefix].pb, input_beam.prob() + p)
                continue
            tail_idx = None if (prefix == '') else char_set.index(prefix[-1])
                # tail_idx = None if (prefix == '') else prefix
            n_prefix = prefix + char_set[idx]
            if n_prefix not in gen_beams:
                gen_beams[n_prefix] = Beam(prefix=n_prefix, pb=NEG_INF, pnb=NEG_INF)
            if idx != tail_idx:
                gen_beams[n_prefix].pnb = np.logaddexp(gen_beams[n_prefix].pnb,
                                                           input_beam.prob() + p)
            else:  # idx == tail_idx
                    # Not merge and include previous pb.
                gen_beams[n_prefix].pnb = np.logaddexp(gen_beams[n_prefix].pnb,
                                                           input_beam.pb + p)
                    # Merge and include previous pnb.
                gen_beams[prefix].pnb = np.logaddexp(gen_beams[prefix].pnb,
                                                         input_beam.pnb + p)
        
    out_beams = gen_beams.values()
    for beam in out_beams:
        sentence = " ".join(char for char in (beam.prefix + suffix))

        ngram_lm_score = kenlm_model.score(sentence)
        beam.plm = ngram_lm_score * lm_panelty + len(beam.prefix) * len_bonus
    out_beams2=sorted(out_beams, key=lambda x: x.total(), reverse=True)

    return out_beams2[:beam_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--saved_model', default="densenet12.pth")
    parser.add_argument('--batch_max_length', type=int, default=150, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=400, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', default=False,help='use rgb input')
    parser.add_argument('--character', type=str, default='', help='character label')
    parser.add_argument('--PAD',default=True, action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--Transformation', type=str, required=False, default="None")
    parser.add_argument('--FeatureExtraction', type=str, required=False, default= "zhiyu_densenet")
    parser.add_argument('--SequenceModeling', type=str, required=False, default="None", help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=False, default="CTC")
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=1088,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--dictfile',default='/home/hjzhan/textrec/charsets/charset7655.txt')
    parser.add_argument('--phase',default='test',help='phase')
    opt = parser.parse_args()
    parser.add_argument('--uselm',default='False')
    opt = parser.parse_args()


    

    opt.FeatureExtraction = "svtr_min"
    opt.saved_model = "/home/hjzhan/zhiyu-final/zhiyu-jiaofu/saved_models/None-svtr_min-None-CTC-Seed30472023-01-05-09:34:11/epoch-0.pth"

    opt.output_channel = 192
    opt.input_channel = 3
    opt.rgb = True

    list1=[]
    lines = open(opt.dictfile,'r').readlines()
    for line in lines:
        list1.append(line.replace('\n', ''))  
    char_set = ['[CTCblank]'] + list1
    opt.character=opt.character+''.join(list1)
    print(len(opt.character))  

    cudnn.benchmark = True
    cudnn.deterministic = True

    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)



    

    model.eval()
    imgpathb = "test_ch.jpg"
    rootpath = "/data/zhiyu/project_test1/"
    savename = "svtr_min_3"
    
    # savename = "densenet8-test"
    opt.uselm = False

    with torch.no_grad():
        batch_size = 1
        t1 = time.time()
        start_time1 = time.time()
        allcount = 0
        flag = True
        for testset in os.listdir(rootpath):
            # if testset != "test_20" and flag:
            #     continue
            flag = False
            count = 0
            count_lmtrue = 0
            start_time = time.time()
            for file in os.listdir(rootpath+testset):
                if not file.endswith("txt"):
                    # print(file)
                    imgpath = os.path.join(rootpath,testset,file)
                    savepath = os.path.join(testset,file)
                    labelfile = file.split(".")[0]+".txt"
                    if os.path.exists(os.path.join(rootpath,testset,labelfile)):
                        try:
                            label = open(os.path.join(rootpath,testset,labelfile)).readlines()[0].replace("\n","")
                            print(count)
                            count+=1
                            allcount+=1
                                
                        except:
                            continue
                        
                    
            
                        img = Image.open(imgpath).convert("RGB")
                        w, h = img.size
                        ratio = w / float(h)

                        resized_w = math.ceil(32 * ratio)
                        padsize = 1600
                        # if resized_w % 800!=0:
                        #     tmp = resized_w % 800
                        #     padsize += (800-tmp)
                        if resized_w >1600:
                            resized_w = 1600
                            # padsize = resized_w
                        transformer = dataset.NormalizePAD((3,32,padsize))
                        resized_image = img.resize((resized_w, 32), Image.BICUBIC)
                        img_tensor = transformer(resized_image)
                        
                            
                        img_tensor = img_tensor.view(1,*img_tensor.size())
                        
                        if 'CTC' in opt.Prediction:
                            text_for_pred = ""
                            preds = model(img_tensor, text_for_pred)
                            pred_lm = ""
                            if opt.uselm:
                                _,preds_lm = ctc_bm_decode(preds=preds)
                                pred_lm = preds_lm[0]
                            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                            _, preds_index = preds.max(2)
                            preds_str = converter.decode(preds_index.data, preds_size.data)[0]
                            
                            
                            
                            
                        print(preds_str,pred_lm)
                        label = label.replace("﻿","")
                        if preds_str == label:
                            acctag = "pure_true"
                        else:
                            acctag = "pure_wrong"
                        if str_normal(preds_str) == str_normal(label):
                            acctag_withnorm = "true_with_norm"
                            count_lmtrue += 1
                        else:
                            acctag_withnorm = "wrong_with_norm"
                        if str_normal(pred_lm) == str_normal(label):
                            acctag_withnorm_withlm = "true_withnormlm"
                        else:
                            acctag_withnorm_withlm = "wrong_withnormlm"
                                
                        lmacc = count_lmtrue / count
                        with open("result/"+savename+".txt","a",encoding="utf-8") as f1: 
                            f1.write(f'{savepath}\t{label}\t{preds_str}\t{pred_lm}\t{acctag}\t{acctag_withnorm}\t{acctag_withnorm_withlm}\t{lmacc}\n')

                                
            end_time = time.time()
            with open("result/"+savename+".txt","a",encoding="utf-8") as f1: 
                f1.write(f'total time: {end_time-start_time}\n')
                f1.write(f'total number: {count}\n')
                f1.write(f'average time: {(end_time-start_time)/float(count)}\n')
        with open("result/"+savename+".txt","a",encoding="utf-8") as f1: 
                f1.write(f'total time: {end_time-start_time1}\n')
                f1.write(f'total number: {count}\n')
                f1.write(f'average time: {(end_time-start_time1)/float(allcount)}\n')
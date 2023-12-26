import torch.nn as nn
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import RCNN_FeatureExtractor
from modules.densenet import DenseNet
from modules.svtrNet import SVTRNet
from modules.edgevit import EdgeViT
from modules.effnetv1 import EfficientNet
from modules.efficientformer_v2 import efficientformerv2_s0, efficientformerv2_s1
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

class Model(nn.Module):
    
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'RCNN':     # 0.47M
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel) 
        elif opt.FeatureExtraction == 'DenseNet': # 0.55M
            self.FeatureExtraction = DenseNet(opt.input_channel, opt.output_channel)
            
        elif opt.FeatureExtraction == "SVTR-T": # 4.16M
            self.FeatureExtraction = SVTRNet(in_channels=opt.input_channel, out_channels=opt.output_channel)
        elif opt.FeatureExtraction == "SVTR-S": # 8.46M
            self.FeatureExtraction = SVTRNet(in_channels=opt.input_channel, out_channels=opt.output_channel,
                                             embed_dim=[96,192,256], depth=[3,6,6], num_heads=[3,6,8],
                                             mixer=['Local']*8 + ['Global']*7)

        elif opt.FeatureExtraction == "EdgeViT-XXS": # 5.32M
            self.FeatureExtraction = EdgeViT(opt.input_channel, opt.output_channel) 
        elif opt.FeatureExtraction == "EdgeViT-XS":  # 9.57M
            self.FeatureExtraction = EdgeViT(opt.input_channel, opt.output_channel,
                                             channels=(48, 96, 240, 384), blocks=(1, 1, 2, 2), heads=(1, 2, 4, 8))  
        # elif opt.FeatureExtraction == "EdgeViT-S": # 11.1M
        #     opt.output_channel = 512
        #     self.FeatureExtraction = EdgeViT(opt.input_channel, opt.output_channel,
        #                                      channels=(48, 96, 240, 384), blocks=(1, 2, 3, 2), heads=(1, 2, 4, 8))
            
        elif opt.FeatureExtraction == "EfficientFormerV2-S0": # 3.6M
            self.FeatureExtraction = efficientformerv2_s0(opt.input_channel, opt.output_channel) 
        elif opt.FeatureExtraction == "EfficientFormerV2-S1": # 6.18M
            self.FeatureExtraction = efficientformerv2_s1(opt.input_channel, opt.output_channel) 
            
        elif opt.FeatureExtraction == "EfficientNet-b0":     # 5.28M     
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b0', opt.input_channel, opt.output_channel)   
        elif opt.FeatureExtraction == "EfficientNet-b1":     # 7.79M 
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b1', opt.input_channel, opt.output_channel)     
        elif opt.FeatureExtraction == "EfficientNet-b2":     # 9.10M   
            self.FeatureExtraction = EfficientNet.from_name('efficientnet-b2', opt.input_channel, opt.output_channel)      
            
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')
            
        """ multi task learning """  
        if opt.multi_task:
            self.GAP = nn.AdaptiveAvgPool1d(1)
            self.Flatten = nn.Flatten()
            self.blur_dec= nn.Linear(self.opt.output_channel, 5, bias=False)   
            
        """ Contrastive learning """   
        if opt.CL:
            self.fc1 = nn.Linear(self.opt.output_channel, self.opt.output_channel, bias=False)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.opt.output_channel, self.opt.z_channel, bias=False)
   

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        # print(input.shape)
        visual_feature = self.FeatureExtraction(input) # h_i
        
        # neck
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3) # [b, w, c]
        
        idx = None
        if self.opt.multi_task:
            # blur recognition branch
            idx = self.blur_dec(self.Flatten(self.GAP(visual_feature.permute(0,2,1)))) 
        
        z = None
        if self.opt.CL:
            # blur recognition branch
            visual_feature = self.fc1(visual_feature)
            visual_feature = self.relu(visual_feature)
            z              = self.fc2(visual_feature)
        
        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        # print(prediction.shape)
        return prediction, z, idx

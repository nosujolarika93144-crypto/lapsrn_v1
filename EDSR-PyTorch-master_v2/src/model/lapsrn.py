# from model import common
# import torch
# import torch.nn as nn

# def make_model(args, parent=False):
#     return LapSRN(args)

# class FeatureExtraction(nn.Module):
#     def __init__(self, conv, n_feats, n_resblocks, act):
#         super(FeatureExtraction, self).__init__()
#         modules_body = [
#             common.RCAB_CBAM(
#                 conv, n_feats, 3, act=act, res_scale=1
#             ) for _ in range(n_resblocks)
#         ]
#         self.body = nn.Sequential(*modules_body)

#     def forward(self, x):
#         return self.body(x)

# class UpsampleBlock(nn.Module):
#     def __init__(self, conv, n_feats, n_resblocks, act, n_colors):
#         super(UpsampleBlock, self).__init__()
#         self.upsample = common.Upsampler(conv, 2, n_feats, act=False) # [最终修正] 确保上采样后不跟多余的激活函数
#         self.reconstruction = nn.Sequential(
#             *[common.RCAB_CBAM(conv, n_feats, 3, act=act) for _ in range(n_resblocks)]
#         )
#         self.output = conv(n_feats, n_colors, 3)

#     def forward(self, feature_from_previous_stage, feature_from_current_lr=None):
#         feature_up = self.upsample(feature_from_previous_stage)
        
#         if feature_from_current_lr is not None:
#              feature_fused = feature_up + feature_from_current_lr
#         else:
#              feature_fused = feature_up
        
#         feature_reconstructed = self.reconstruction(feature_fused)
#         image_output = self.output(feature_reconstructed)
        
#         return feature_reconstructed, image_output

# class LapSRN(nn.Module):
#     def __init__(self, args, conv=common.default_conv):
#         super(LapSRN, self).__init__()
        
#         n_resblocks = args.n_resblocks
#         n_feats = args.n_feats
#         n_colors = args.n_colors
        
#         # [最终修正] 使用LeakyReLU以处理归一化后的负值数据
#         act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
#         self.head_x8 = conv(n_colors, n_feats, 3)
#         self.head_x4 = conv(n_colors, n_feats, 3)
#         self.head_x2 = conv(n_colors, n_feats, 3)

#         self.feature_extractor_x4 = FeatureExtraction(conv, n_feats, n_resblocks, act)
#         self.feature_extractor_x2 = FeatureExtraction(conv, n_feats, n_resblocks, act)

#         self.upsample_block_x4 = UpsampleBlock(conv, n_feats, n_resblocks, act, n_colors)
#         self.upsample_block_x2 = UpsampleBlock(conv, n_feats, n_resblocks, act, n_colors)
#         self.upsample_block_x1 = UpsampleBlock(conv, n_feats, n_resblocks, act, n_colors)

#     def forward(self, lr_x8, lr_x4, lr_x2):
#         feat_x8 = self.head_x8(lr_x8)
#         feat_x4_in = self.feature_extractor_x4(self.head_x4(lr_x4))
#         feat_x4_out, img_out_x4 = self.upsample_block_x4(feat_x8, feat_x4_in)

#         feat_x2_in = self.feature_extractor_x2(self.head_x2(lr_x2))
#         feat_x2_out, img_out_x2 = self.upsample_block_x2(feat_x4_out, feat_x2_in)

#         _ , img_out_x1 = self.upsample_block_x1(feat_x2_out)

#         return [img_out_x4, img_out_x2, img_out_x1]
from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return LapSRN(args)

class FeatureExtraction(nn.Module):
    def __init__(self, conv, n_feats, n_resblocks, act, reduction):
        super(FeatureExtraction, self).__init__()
        # [最终修正] 使用 RCAB 替换 RCAB_CBAM
        modules_body = [
            common.RCAB(
                conv, n_feats, 3, reduction, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)

class UpsampleBlock(nn.Module):
    def __init__(self, conv, n_feats, n_resblocks, act, reduction, n_colors):
        super(UpsampleBlock, self).__init__()
        self.upsample = common.Upsampler(conv, 2, n_feats, act=False)
        # [最终修正] 使用 RCAB 替换 RCAB_CBAM
        self.reconstruction = nn.Sequential(
            *[common.RCAB(conv, n_feats, 3, reduction, act=act) for _ in range(n_resblocks)]
        )
        self.output = conv(n_feats, n_colors, 3)

    def forward(self, feature_from_previous_stage, feature_from_current_lr=None):
        feature_up = self.upsample(feature_from_previous_stage)
        
        if feature_from_current_lr is not None:
             feature_fused = feature_up + feature_from_current_lr
        else:
             feature_fused = feature_up
        
        feature_reconstructed = self.reconstruction(feature_fused)
        image_output = self.output(feature_reconstructed)
        
        return feature_reconstructed, image_output

class LapSRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LapSRN, self).__init__()
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        n_colors = args.n_colors
        reduction = args.reduction # 从args中获取reduction参数
        
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.head_x8 = conv(n_colors, n_feats, 3)
        self.head_x4 = conv(n_colors, n_feats, 3)
        self.head_x2 = conv(n_colors, n_feats, 3)

        self.feature_extractor_x4 = FeatureExtraction(conv, n_feats, n_resblocks, act, reduction)
        self.feature_extractor_x2 = FeatureExtraction(conv, n_feats, n_resblocks, act, reduction)

        self.upsample_block_x4 = UpsampleBlock(conv, n_feats, n_resblocks, act, reduction, n_colors)
        self.upsample_block_x2 = UpsampleBlock(conv, n_feats, n_resblocks, act, reduction, n_colors)
        self.upsample_block_x1 = UpsampleBlock(conv, n_feats, n_resblocks, act, reduction, n_colors)

    def forward(self, lr_x8, lr_x4, lr_x2):
        feat_x8 = self.head_x8(lr_x8)
        feat_x4_in = self.feature_extractor_x4(self.head_x4(lr_x4))
        feat_x4_out, img_out_x4 = self.upsample_block_x4(feat_x8, feat_x4_in)

        feat_x2_in = self.feature_extractor_x2(self.head_x2(lr_x2))
        feat_x2_out, img_out_x2 = self.upsample_block_x2(feat_x4_out, feat_x2_in)

        _ , img_out_x1 = self.upsample_block_x1(feat_x2_out)

        return [img_out_x4, img_out_x2, img_out_x1]
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from .mobilefacenet import MobileFaceNet


class SCLFER(nn.Module):
    def __init__(self, num_class=7, num_head=4, pretrained=True):
        super(SCLFER, self).__init__()
        
        # MobileFaceNet을 백본으로 사용
        self.face_backbone = MobileFaceNet([112, 112], 136)
        
        # 사전 학습된 가중치 로딩
        if pretrained:
            face_backbone_checkpoint = torch.load('C:/Users/steve/OneDrive/Desktop/SCLFER/models/pretrain/mobilefacenet_model_best.pth.tar', 
                                     map_location=lambda storage, loc: storage)
            self.face_backbone.load_state_dict(face_backbone_checkpoint['state_dict'])
        
        # 기존 모델 코드 유지하되 ChannelAttentionHead 사용
        self.num_head = num_head
        for i in range(num_head):
            setattr(self, "cat_head%d" % i, ChannelAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)
        
        self.heads_sum_layer = nn.Identity()
    
    def forward(self, x):
        # 입력 이미지를 MobileFaceNet에 맞게 크기 조정 (112x112)
        x_face = F.interpolate(x, size=112)
        
        # MobileFaceNet으로 특징 추출
        _, features = self.face_backbone(x_face)  # features shape: [B, 512, 7, 7]
        
        # Channel Attention만 사용하는 forward 로직
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, "cat_head%d" % i)(features))
        
        heads = torch.stack(heads).permute([1, 0, 2])
        if heads.size(1) > 1:
            heads = F.log_softmax(heads, dim=1)
        
        heads_sum = heads.sum(dim=1)
        heads_sum = self.heads_sum_layer(heads_sum.view(heads_sum.size(0), -1, 1, 1))
        
        out = self.fc(heads_sum.view(heads_sum.size(0), -1))
        out = self.bn(out)
   
        return out, features, heads


# ChannelAttentionHead만 사용 (SpatialAttention 제거)
class ChannelAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ca = ChannelAttention()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # Apply channel attention directly without spatial attention
        ca = self.ca(x)
        return ca


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )

    def forward(self, x):
        # Modified to take x directly instead of sa
        features = self.gap(x)
        features = features.view(features.size(0), -1)
        y = self.attention(features)
        out = features * y
        
        return out
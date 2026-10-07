import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import warnings
from einops import rearrange

class MLFE(nn.Module):
    def __init__(self, in_channels=9, pretrained=True):
        super().__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
        resnet = resnet50(weights=weights)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = nn.Identity()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x);
        x = self.maxpool(x)
        f1 = self.layer1(x);
        f2 = self.layer2(f1);
        f3 = self.layer3(f2)
        return f1, f2, f3


class FreqEncoder(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False), nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_freq = torch.fft.fft2(x.to(torch.float32), norm='ortho')
        x_freq_abs = torch.abs(x_freq)
        freq_feature = self.freq_cnn(x_freq_abs)
        return freq_feature


# --- BGCF ---
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x_q, x_kv):
        q = self.to_q(x_q)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class BGCF(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        # forward_fusion: Freq -> Spatial
        self.norm_s_in = nn.LayerNorm(dim)
        self.norm_f_in1 = nn.LayerNorm(dim)
        self.attn_s_out = CrossAttention(dim, heads, dim_head, dropout)
        # backward_fusion: Spatial -> Freq
        self.norm_f_in2 = nn.LayerNorm(dim)
        self.norm_s_in2 = nn.LayerNorm(dim)
        self.attn_f_out = CrossAttention(dim, heads, dim_head, dropout)

        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, x_spatial, x_freq):
        B, C, H, W = x_spatial.shape
        x_s_flat = rearrange(x_spatial, 'b c h w -> b (h w) c')
        x_f_flat = rearrange(x_freq, 'b c h w -> b (h w) c')

        q_f = self.norm_f_in1(x_f_flat)
        kv_s = self.norm_s_in(x_s_flat)
        calibrated_s = self.attn_s_out(q_f, kv_s) + x_s_flat

        q_s = self.norm_s_in2(x_s_flat)
        kv_f = self.norm_f_in2(x_f_flat)
        focused_f = self.attn_f_out(q_s, kv_f) + x_f_flat

        g = self.gate(torch.cat((calibrated_s, focused_f), dim=-1))
        fused_flat = g * calibrated_s + (1 - g) * focused_f

        return rearrange(fused_flat, 'b (h w) c -> b c h w', h=H, w=W)


# --- CMFT ---
class CMFT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, num_modalities=4, dropout=0.):
        super().__init__()
        self.modal_embeddings = nn.Parameter(torch.randn(1, num_modalities, dim))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        self.num_modalities = num_modalities

    def forward(self, feature_list):
        base_h, base_w = feature_list[0].shape[-2:]
        processed_feats = []
        for feat in feature_list:
            if feat.shape[-2:] != (base_h, base_w):
                feat = F.interpolate(feat, size=(base_h, base_w), mode='bilinear', align_corners=False)
            processed_feats.append(rearrange(feat, 'b c h w -> b (h w) c'))

        
        token_sequences = []
        for i, seq in enumerate(processed_feats):
            token_sequences.append(seq + self.modal_embeddings[:, i, :])

       
        full_sequence = torch.cat(token_sequences, dim=1)

        transformed_sequence = self.transformer(full_sequence)

        chunked_sequences = torch.chunk(transformed_sequence, self.num_modalities, dim=1)
        output_maps = [rearrange(chunk, 'b (h w) c -> b c h w', h=base_h, w=base_w) for chunk in chunked_sequences]

        return torch.cat(output_maps, dim=1)


# --- SF-GANet (Spatial-Frequency Gated Attention Network) ---
class SF_GANet(nn.Module):

    def __init__(self, num_classes, in_channels=9, pretrained=True, supcon=False):
        super().__init__()
        self.supcon = supcon

        # 1. backbone
        self.backbone = MLFE(in_channels=in_channels, pretrained=pretrained)

        self.freq_encoder = FreqEncoder(channels=256)

        self.proj_f1 = nn.Conv2d(256, 256, 1)
        self.proj_f2 = nn.Conv2d(512, 256, 1)
        self.proj_f3 = nn.Conv2d(1024, 256, 1)

        # 2. BGCF
        self.fusion1 = BGCF(dim=256, heads=8, dim_head=32)
        self.fusion2 = BGCF(dim=256, heads=8, dim_head=32)
        self.fusion3 = BGCF(dim=256, heads=8, dim_head=32)

        # 3. CMFT
        self.enhancer = CMFT(dim=256, depth=2, heads=8, mlp_dim=512, num_modalities=4)

        feature_dim = 1024

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x):
        f1, f2, f3 = self.backbone(x)
        f_freq = self.freq_encoder(f1) 

        f1_proj, f2_proj, f3_proj = self.proj_f1(f1), self.proj_f2(f2), self.proj_f3(f3)


        f_fused1 = self.fusion1(f1_proj, F.interpolate(f_freq, size=f1_proj.shape[-2:], mode='bilinear'))
        f_fused2 = self.fusion2(f2_proj, F.interpolate(f_freq, size=f2_proj.shape[-2:], mode='bilinear'))
        f_fused3 = self.fusion3(f3_proj, F.interpolate(f_freq, size=f3_proj.shape[-2:], mode='bilinear'))

        base_size = f1_proj.shape[-2:]
        feature_list_for_cmft = [
            f_fused1,
            F.interpolate(f_fused2, size=base_size, mode='bilinear'),
            F.interpolate(f_fused3, size=base_size, mode='bilinear'),
            F.interpolate(f_freq, size=base_size, mode='bilinear')
        ]

        enhanced_feat = self.enhancer(feature_list_for_cmft)

        pooled = self.gap(enhanced_feat)
        features = torch.flatten(pooled, 1)
        logits = self.classifier(features)

        if self.supcon:
            return features, logits
        else:
            return logits



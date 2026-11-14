"""Lightweight geometry-aware PCT baseline for point segmentation."""
import torch
import torch.nn as nn

class OffsetAttention(nn.Module):
    def __init__(self, channels):
        super().__init__(); reduced = max(channels // 4, 1); self.scale = reduced ** -.5
        self.q = nn.Conv1d(channels, reduced, 1, bias=False); self.k = nn.Conv1d(channels, reduced, 1, bias=False)
        self.v = nn.Conv1d(channels, channels, 1)
        self.offset = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.BatchNorm1d(channels), nn.ReLU())
    def forward(self, x):
        attention = torch.softmax(torch.bmm(self.q(x).transpose(1,2), self.k(x)) * self.scale, dim=-1)
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-6)
        return x + self.offset(x - torch.bmm(self.v(x), attention))

class GeometryAwarePCT(nn.Module):
    def __init__(self, num_classes=31, in_channels=9, width=96):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv1d(in_channels,64,1),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,width,1),nn.BatchNorm1d(width),nn.ReLU())
        self.attn1, self.attn2 = OffsetAttention(width), OffsetAttention(width)
        self.fuse = nn.Sequential(nn.Conv1d(width*2,256,1),nn.BatchNorm1d(256),nn.LeakyReLU(.2))
        self.head = nn.Sequential(nn.Conv1d(512,256,1),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.3),nn.Conv1d(256,128,1),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,num_classes,1))
    def forward(self, features):
        x = self.embed(features.transpose(1,2)); x1 = self.attn1(x); x2 = self.attn2(x1)
        local = self.fuse(torch.cat((x1,x2),1)); global_feature = local.max(2,keepdim=True).values.expand_as(local)
        return self.head(torch.cat((local,global_feature),1)).transpose(1,2)

PointCloudTransformer = GeometryAwarePCT
if __name__ == "__main__":
    model=GeometryAwarePCT(); output=model(torch.randn(2,1024,9)); output.mean().backward(); print("OK", output.shape, sum(p.numel() for p in model.parameters()))

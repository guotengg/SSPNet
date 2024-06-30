import torch
import torch.nn as nn
import torch.nn.functional as F


# M=1 K=*
class PLEs(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.k = k
        self.out_ch = out_ch
        # self.x = x
        # self.y = y
        offset_list = []
        for x in range(k):
            conv = nn.Conv2d(in_ch, 2, 1, 1, 0, bias=True)
            offset_list.append(conv)
        self.offset_conv = nn.ModuleList(offset_list)
        self.weight_conv = nn.Sequential(nn.Conv2d(in_ch, k, 1, 1, 0, bias=True), nn.Softmax(1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        b, c, h, w = input.size()
        proj_feat = self.conv(input)
        offsets = []
        for x in range(self.k):
            flow = self.offset_conv[x](input)
            offsets.append(flow)
        offsetweights = torch.repeat_interleave(self.weight_conv(input), self.out_ch, 1)
        feats = []
        for x in range(self.k):
            flow = offsets[x]
            flow = flow.permute(0, 2, 3, 1)
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            grid = torch.stack((grid_x, grid_y), 2).float()
            grid.requires_grad = False
            grid = grid.type_as(proj_feat)
            vgrid = grid + flow
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            feat = F.grid_sample(proj_feat, vgrid_scaled, mode='bilinear', padding_mode='zeros')
            feats.append(feat)
        feat = torch.cat(feats, 1) * offsetweights
        feat = sum(torch.split(feat, self.out_ch, 1))
        return feat


class PLEs_bad(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.k = k
        self.out_ch = out_ch
        offset_list = []
        for x in range(k):
            conv = nn.Conv2d(in_ch, 2, 1, 1, 0, bias=True)
            offset_list.append(conv)
        self.offset_conv = nn.ModuleList(offset_list)
        self.weight_conv = nn.Sequential(nn.Conv2d(in_ch, k, 1, 1, 0, bias=True), nn.Softmax(1))
        # self.points = keypoints
        # print()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, input, ax, ay):
        b, c, h, w = input.size()
        proj_feat = self.conv(input)
        offsets = []
        for x in range(self.k):
            flow = self.offset_conv[x](input)
            offsets.append(flow)
        offsetweights = torch.repeat_interleave(self.weight_conv(input), self.out_ch, 1)
        feats = []
        for x in range(self.k):
            flow = offsets[x]
            flow = flow.permute(0, 2, 3, 1)
            # grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            grid_y = ay
            grid_x = ax
            # ax = ax.trunc().clone().type(torch.long)
            # ay = ay.trunc().clone().type(torch.long)
            feat_num = []
            for num in range(input.size()[0]):
                grid = torch.stack((grid_x[num, :], grid_y[num, :]), 1).float()
                index = grid.trunc().clone().type(torch.long)
                grid.requires_grad = False
                grid = grid.type_as(proj_feat[num, :, :, :])
                flows = []
                for id in index:
                    flow_ = flow[num, id[1], id[0], :]
                    flows.append(flow_)
                flow = torch.stack(flows)
                vgrid = grid + flow
                vgrid_x = 2.0 * vgrid[:, 0] / max(w - 1, 1) - 1.0
                vgrid_y = 2.0 * vgrid[:, 1] / max(h - 1, 1) - 1.0
                vgrid_scaled = torch.stack((vgrid_x, vgrid_y))
                feat = F.grid_sample(proj_feat[num], vgrid_scaled, mode='bilinear', padding_mode='zeros')
                feat_num.append(feat)
                feat_nums = torch.cat(feat_num, 0)
            feats.append(feat_nums)
        feat = torch.cat(feats, 1) * offsetweights
        feat = sum(torch.split(feat, self.out_ch, 1))
        return feat


class PLEk(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.k = k
        self.out_ch = out_ch
        offset_list = []
        for x in range(k):
            conv = nn.Conv2d(in_ch, 2, 1, 1, 0, bias=True)
            offset_list.append(conv)
        self.offset_conv = nn.ModuleList(offset_list)
        self.weight_conv = nn.Sequential(nn.Conv2d(in_ch, k, 1, 1, 0, bias=True), nn.Softmax(1))
        # self.points = keypoints
        # print()
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, input, ax, ay):
        b, c, h, w = input.size()
        proj_feat = self.conv(input)
        offsets = []
        for x in range(self.k):
            flow = self.offset_conv[x](input)
            offsets.append(flow)
        offsetweights = torch.repeat_interleave(self.weight_conv(input), self.out_ch, 1)
        feats = []
        for x in range(self.k):
            flow = offsets[x]
            flow = flow.permute(0, 2, 3, 1)
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
            # g_y = ay
            # g_x = ax
            grid = torch.stack((grid_x, grid_y), 2).float()
            grid.requires_grad = False
            grid = grid.type_as(proj_feat)
            vgrid = grid + flow
            vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            feat = F.grid_sample(proj_feat, vgrid_scaled, mode='bilinear', padding_mode='zeros')
            feats.append(feat)
        feat = torch.cat(feats, 1) * offsetweights
        feat_nums = []
        index = torch.stack((ax, ay), 2)
        index = index.trunc().clone().type(torch.long)
        for num in range(input.size()[0]):
            feat_ids = []
            for id in index[num]:
                feat_id = feat[num, :, id[1], id[0]]
                feat_ids.append(feat_id)
            feat_ids = torch.stack(feat_ids)
            feat_nums.append(feat_ids)
        feats = torch.stack(feat_nums)
        feat = sum(torch.split(feats, self.out_ch, 2))
        return feat


def sparse(feature_map, ratio=0.75):
    # feature map shape
    b, c, h, w = feature_map.shape

    # point number after sampling
    sample_h = int(h * ratio)
    sample_w = int(w * ratio)

    # sampling on H and W
    indices_h = torch.linspace(0, h - 1, sample_h).round().long().to(feature_map.device)
    indices_w = torch.linspace(0, w - 1, sample_w).round().long().to(feature_map.device)

    # sampling on feature map
    sampled_feature_map = feature_map[:, :, indices_h, :]
    sampled_feature_map = sampled_feature_map[:, :, :, indices_w]

    return sampled_feature_map

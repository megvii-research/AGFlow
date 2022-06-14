import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AGRDecoder(nn.Module):
    def __init__(self, args, chnn=128, chnn_in=128):
        super().__init__()
        self.args = args
        cor_planes = 4 * (2 * args.corr_radius + 1) ** 2
        self.C_cor = nn.Sequential(
            nn.Conv2d(cor_planes, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, 3, padding=1),
            nn.ReLU(inplace=True))
        self.C_flo = nn.Sequential(
            nn.Conv2d(2, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))
        self.C_mo = nn.Sequential(
            nn.Conv2d(192+64, 128-2, 3, padding=1),
            nn.ReLU(inplace=True))

        self.graph = AGR(args, 128)
        self.gru = SepConvGRU(hidden_dim=chnn, input_dim=128+chnn)
        self.C_flow = nn.Sequential(
            nn.Conv2d(chnn, chnn*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn*2, 2, 3, padding=1))
        self.C_mask = nn.Sequential(
            nn.Conv2d(chnn, chnn*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn*2, 64*9, 1, padding=0))
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _mo_enc(self, flow, corr):
        feat_cor = self.C_cor(corr)
        feat_flo = self.C_flo(flow)
        feat_cat = torch.cat([feat_cor, feat_flo], dim=1)
        feat_mo = self.C_mo(feat_cat)
        feat_mo = torch.cat([feat_mo, flow], dim=1)
        return feat_mo

    def forward(self, net, inp, corr, flow, itr, upsample=True):
        feat_mo = self._mo_enc(flow, corr)
        feats = self.graph(inp, feat_mo, itr)
        inp = torch.cat(feats, dim=1)
        net = self.gru(net, inp)
        delta_flow = self.C_flow(net)

        # scale mask to balence gradients
        mask = .25 * self.C_mask(net)
        return net, mask, delta_flow, self.zero


class AGR(nn.Module):
    def __init__(self, args, chnn, k=128):
        super().__init__()
        self.C_cpr = nn.Sequential(
            nn.Conv2d(chnn, chnn, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn, k, 1))
        self.C_cad = nn.Conv1d(chnn, chnn, kernel_size=1)
        self.gcn = nn.ModuleList([GCN(chnn), GCN(chnn)])
        self.agcn = AGCN(chnn, k)
        self.C_mpr = nn.Conv2d(chnn, k, kernel_size=1)
        self.C_ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chnn, chnn//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn//4, chnn, 1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.zero = nn.Parameter(torch.zeros(1), requires_grad=False)

    def __func_cpr(self, x, C_cpr):
        b, c, h, w = x.shape
        mp = C_cpr(x).view(b, -1, h*w)
        vs = torch.einsum('b c n , b k n -> b c k', x.view(b, c, -1), mp)
        vs = F.normalize(vs, p=2, dim=1) 
        return vs, mp

    def __func_cg(self, vc, C_cad, gcn):
        vca = F.normalize(C_cad(vc), p=2, dim=1)
        A = torch.einsum('b c k , b c l -> b k l', vca, vca)
        vc_T = vc.permute(0, 2, 1).contiguous()
        vco = F.relu_(gcn[0](vc_T, A))
        vco = (gcn[1](vco, A)).permute(0, 2, 1).contiguous()
        return vco

    def __func_mpr(self, x, C_mpr):
        b, c, h, w = x.shape
        mp = C_mpr(x).view(b, -1, h*w)
        mp = F.softmax(mp, dim=-1)
        vs = torch.einsum('b c n , b k n -> b c k', x.view(b, c, -1), mp)
        vs = F.normalize(vs, p=2, dim=1)
        return vs

    def forward(self, *inputs):
        feat_ctx, feat_mo, itr = inputs
        b, c, h, w = feat_ctx.shape

        if itr == 0:
            vc, mp = self.__func_cpr(feat_ctx, self.C_cpr)
            vco = self.__func_cg(vc, self.C_cad, self.gcn)
            feat_ctxa = torch.bmm(vco, mp)
            feat_ctx = feat_ctx + feat_ctxa.view(b, -1, h, w) * self.alpha
            self.vc, self.mp, self.feat_ctx = vc, mp, feat_ctx

        vm = self.__func_mpr(feat_mo, self.C_mpr)
        vmo = self.agcn(self.vc, vm)
        feat_moa = torch.bmm(vmo, self.mp)
        feat_mo = feat_mo + feat_moa.view(b, -1, h, w) * self.beta
        feat_ctx = self.feat_ctx + self.feat_ctx * torch.sigmoid(self.C_ca(feat_mo))
        return feat_ctx, feat_mo


class GCN(nn.Module):
    def __init__(self, chnn):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(chnn, chnn))
        self.bias = nn.Parameter(torch.Tensor(chnn))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, A, sm=True):
        b = x.shape[0]
        x = x.matmul(self.weight.repeat(b, 1, 1))
        if self.bias is not None:
            x = x + self.bias.repeat(b, 1, 1)
        if sm:
            A = F.softmax(A, dim=-1)
        x = A.matmul(x)
        return x


class AGCN(nn.Module):
    def __init__(self, chnn, k):
        super().__init__()
        self.C_ak = nn.Conv1d(chnn, k, 1)
        self.C_c = nn.Conv1d(chnn, chnn, 1)
        self.C_mad = nn.Conv1d(chnn, chnn, kernel_size=1)
        self.gcn = GCN(chnn)

    def forward(self, *inputs):
        vc, vm = inputs
        vcw = self.C_ak(vc)
        vcw = F.softmax(vcw, dim=1)  
        w = vcw.squeeze(0).permute(1, 0).contiguous()
        vm = F.relu_(self.C_c(vm))
        vma = F.linear(vm, w)
        vma = self.C_mad(vma)
        A = torch.einsum('b c k , b c l -> b k l', vma, vma)
        vc_T = vm.permute(0, 2, 1).contiguous()
        vco = self.gcn(vc_T, A)
        vco = vco.permute(0, 2, 1).contiguous()
        return vco


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


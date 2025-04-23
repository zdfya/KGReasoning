import torch
from torch import nn
import dgl
import dgl.function as fn
import torch.nn.functional as F


class CompGCN(nn.Module):
    def __init__(self, args):
        super(CompGCN, self).__init__()
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.GraphCov = None
        self.init_model()

    def init_model(self):
        # --------- 基础嵌入 & 图卷积 ----------
        # 实体嵌入 [num_ent, emb_dim]
        self.ent_emb = nn.Parameter(torch.Tensor(self.args.num_ent, self.args.emb_dim))
        # 关系嵌入：正反向共 2*num_rel
        self.rel_emb = nn.Parameter(torch.Tensor(self.args.num_rel * 2, self.args.emb_dim))
        nn.init.xavier_normal_(self.ent_emb, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.GraphCov = CompGCNCov(
            in_channels=self.args.emb_dim,
            out_channels=self.args.emb_dim,     # 保持和 emb_dim 一致，或改用 hidden_dim
            act=torch.tanh,
            bias=False,
            drop_rate=self.args.gc_drop,        # 图卷积 Dropout，可在 args 中设置
            opn=self.args.opn
        )
        self.bias = nn.Parameter(torch.zeros(self.args.num_ent))
        self.node_drop = nn.Dropout(self.args.node_drop)

        # --------- ConvE 解码器部分 ----------
        # Embedding lookup for entities（用于 DistMult 同时也是 ConvE 最后一层 before fc）
        self.emb_ent = nn.Embedding(self.args.num_ent, self.args.emb_dim)

        # Dropout hyperparams
        self.inp_drop  = nn.Dropout(self.args.input_drop)
        self.hid_drop  = nn.Dropout(self.args.hid_drop)
        self.feat_drop = nn.Dropout2d(self.args.feat_drop)

        # 动态卷积层
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.args.num_filt,
            kernel_size=(self.args.ker_sz, self.args.ker_sz),
            stride=1, padding=0, bias=False
        )
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.args.num_filt)

        # ------------- 关键：动态计算 fc 输入维度 -------------
        # 拼接后 feature map 大小： H_in = 2*k_h, W_in = k_w
        H_in = 2 * self.args.k_h
        W_in =     self.args.k_w
        # 卷积输出： H_out = H_in - ker_sz + 1, W_out = W_in - ker_sz + 1
        H_out = H_in - self.args.ker_sz + 1
        W_out = W_in - self.args.ker_sz + 1
        # 全连接输入维度
        fc_in_dim = self.args.num_filt * H_out * W_out

        # 全连接层：映射回 emb_dim 大小
        self.fc = nn.Linear(fc_in_dim, self.args.emb_dim)

        # BatchNorm1d 对齐 emb_dim
        self.bn2 = nn.BatchNorm1d(self.args.emb_dim)

        # 最后一层偏置（可选）
        self.register_parameter('b', nn.Parameter(torch.zeros(self.args.num_ent)))

    def forward(self, graph, relation, norm, triples):
        head, rela = triples[:, 0], triples[:, 1]
        # 图卷积更新
        x, r = self.ent_emb, self.rel_emb
        x, r = self.GraphCov(graph, x, r, relation, norm)
        x = self.node_drop(x)

        head_emb = x[head]
        rela_emb = r[rela]

        if self.args.decoder_model.lower() == 'conve':
            score = self._forward_conve(head_emb, rela_emb, x)
        elif self.args.decoder_model.lower() == 'distmult':
            score = self._forward_distmult(head_emb, rela_emb)
        else:
            raise ValueError("Decoder must be 'conve' or 'distmult'")
        return score

    def calc_loss(self, preds, labels):
        if preds.size(0) != labels.size(0):
            preds = preds[:labels.size(0)]
        return F.binary_cross_entropy(preds, labels)

    def _forward_distmult(self, head_emb, rela_emb):
        obj = head_emb * rela_emb
        x = torch.mm(obj, self.emb_ent.weight.t())
        x = x + self.bias
        return torch.sigmoid(x)

    def _forward_conve(self, sub_emb, rel_emb, all_ent):
        # 1) 拼接并 BN -> Conv -> BN -> ReLU -> Drop2d
        x = self._concat(sub_emb, rel_emb)  # (B,1,2*k_h,k_w)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feat_drop(x)

        # 2) 展平 -> FC -> Drop -> BN -> ReLU
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 3) 与所有实体嵌入相乘 -> 加 bias -> Sigmoid
        x = torch.mm(x, all_ent.t())
        x = x + self.b
        return torch.sigmoid(x)

    def _concat(self, ent_embed, rel_embed):
        # 保证使用 args.k_h/k_w
        B = ent_embed.size(0)
        ent = ent_embed.view(B, 1, 2 * self.args.k_h, self.args.k_w)
        rel = rel_embed.view(B, 1, 2 * self.args.k_h, self.args.k_w)
        return torch.cat([ent, rel], dim=2)


class CompGCNCov(nn.Module):
    """The comp graph convolution layers"""

    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr'):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.opn = opn

        self.in_w = self._param([in_channels, out_channels])
        self.out_w = self._param([in_channels, out_channels])
        self.loop_w = self._param([in_channels, out_channels])
        self.w_rel = self._param([in_channels, out_channels])
        self.loop_rel = self._param([1, in_channels])

        self.drop = nn.Dropout(drop_rate)
        self.bn = nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.rel_wt = None

    def _param(self, shape):
        p = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(p, gain=nn.init.calculate_gain('relu'))
        return p

    def message_func(self, edges: dgl.udf.EdgeBatch):
        et = edges.data['type']
        E = et.numel()
        data = self.comp(edges.src['h'], self.rel[et])
        msg = torch.cat([
            torch.matmul(data[: E // 2], self.in_w),  # 正向
            torch.matmul(data[E // 2 :], self.out_w)   # 反向
        ], dim=0)
        # 正确广播一维 norm
        norm = edges.data['norm'].view(-1)
        msg = msg * norm.unsqueeze(1)
        return {'msg': msg}

    def reduce_func(self, nodes: dgl.udf.NodeBatch):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, e):
        def conj(a): return a.clone().conj()
        def ccorr(a, b):
            return torch.fft.irfft(conj(torch.fft.rfft(a)) * torch.fft.rfft(b), n=a.size(-1))
        if self.opn == 'mult': return h * e
        if self.opn == 'sub' : return h - e
        if self.opn == 'corr': return ccorr(h, e.expand_as(h))
        raise KeyError(f'opn {self.opn} not recognized')

    # 信息更新
    def forward(self, g: dgl.graph, x, rel_repr, etype, norm):
        self.rel = rel_repr
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = etype
        g.edata['norm'] = norm
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        h = g.ndata.pop('h')
        loop = torch.matmul(self.comp(x, self.loop_rel), self.loop_w)
        h = h + loop / 3
        if self.bias is not None:
            h = h + self.bias
        h = self.bn(h)
        return self.act(h), torch.matmul(self.rel, self.w_rel)
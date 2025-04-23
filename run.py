import os
import argparse
import time
import random
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl

from utils import process, TrainDataset, TestDataset
from model.compgcn import CompGCN, CompGCNCov


def load_data(data_dir):
    raw_data = {'train': [], 'valid': [], 'test': []}
    ent2id, rel2id = {}, {}

    # 第一遍：收集实体、关系
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(data_dir, f"{split}.txt")) as f:
            for line in f:
                h, r, t = line.strip().split()
                if h not in ent2id: ent2id[h] = len(ent2id)
                if t not in ent2id: ent2id[t] = len(ent2id)
                if r not in rel2id: rel2id[r] = len(rel2id)

    # 添加反向关系
    base = len(rel2id)
    for r, idx in list(rel2id.items()):
        rel2id[r + '_reverse'] = idx + base

    # 第二遍：填充 raw_data
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(data_dir, f"{split}.txt")) as f:
            for line in f:
                h, r, t = line.strip().split()
                raw_data[split].append((ent2id[h], rel2id[r], ent2id[t]))

    return raw_data, ent2id, rel2id


class Runner:
    def __init__(self, p):
        self.p = p

        # 1) 加载并数值化
        data_path = os.path.join('data', p.dataset)
        raw_data, self.ent2id, self.rel2id = load_data(data_path)
        self.train_data = np.array(raw_data['train'], dtype=int)
        self.valid_data = np.array(raw_data['valid'], dtype=int)
        self.test_data  = np.array(raw_data['test'],  dtype=int)

        # 2) 实体／关系数
        self.num_ent  = len(self.ent2id)
        self.num_rels = len(self.rel2id) // 2

        # 3) 回写给 CompGCN.init_model
        p.num_ent        = self.num_ent
        p.num_rel        = self.num_rels
        p.emb_dim        = p.embed_dim if p.embed_dim is not None else (p.k_w * p.k_h)
        p.inp_drop       = p.input_drop
        p.hid_drop       = p.hid_drop
        p.fet_drop       = p.feat_drop
        p.decoder_model  = p.decoder_model or p.score_func
        p.out_dim        = p.out_dim or p.emb_dim

        # 4) 构造带标签 triplets
        self.triplets = process(raw_data, self.num_rels)

        # 5) 设备
        self.device = torch.device(f'cuda:{p.gpu}') if p.gpu>=0 and torch.cuda.is_available() else torch.device('cpu')

        # 6) DataLoader
        self.data_iter = self.get_data_iter()

        # 7) 构建 DGL 图
        self.g = self.build_graph().to(self.device)

        # 8) 边类型 & 归一化
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()

        # 9) 模型 & 优化器
        self.model = CompGCN(p).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=p.lr, weight_decay=p.l2)

        # 记录最佳指标
        self.best_val_mrr = 0.
        self.best_epoch   = 0
        self.best_val_results = {}

    def build_graph(self):
        src, dst = [], []
        for h, r, t in self.train_data:
            src += [h, t]; dst += [t, h]
        return dgl.graph((src, dst), num_nodes=self.num_ent)

    def get_edge_dir_and_norm(self):
        deg = self.g.in_degrees().float().clamp(min=1)
        norm = 1.0 / deg
        edge_norm = norm[self.g.edges()[1]]
        rels = np.concatenate([self.train_data[:,1], self.train_data[:,1] + self.num_rels])
        edge_type = torch.tensor(rels, dtype=torch.long, device=self.device)
        return edge_type, edge_norm.unsqueeze(1).to(self.device)

    def get_data_iter(self):
        def loader(split):
            cls = TrainDataset if split=='train' else TestDataset
            return DataLoader(
                cls(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=(split=='train'),
                num_workers=self.p.num_workers
            )
        return {
            'train':      loader('train'),
            'valid_tail': loader('valid_tail'),
            'valid_head': loader('valid_head'),
            'test_tail':  loader('test_tail'),
            'test_head':  loader('test_head'),
        }

    def train(self):
        self.model.train()
        total, cnt = 0.0, 0
        for triples, labels in self.data_iter['train']:
            triples, labels = triples.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(self.g, self.edge_type, self.edge_norm, triples)
            loss  = self.model.calc_loss(preds, labels)
            loss.backward(); self.optimizer.step()
            total += loss.item() * triples.size(0)
            cnt   += triples.size(0)
        return total / cnt

    def _predict(self, split, mode):
        results = {'count':0,'mr':0,'mrr':0,'hits@1':0,'hits@3':0,'hits@10':0}
        self.model.eval()
        with torch.no_grad():
            for triples, labels in self.data_iter[f'{split}_{mode}']:
                triples, labels = triples.to(self.device), labels.to(self.device)
                scores = self.model(self.g, self.edge_type, self.edge_norm, triples)
                if scores.size(0) != triples.size(0):
                    scores = scores[:triples.size(0)]
                b   = torch.arange(scores.size(0), device=self.device)
                obj = triples[:,2]
                tgt = scores[b, obj]
                mask = labels.bool()
                scores[mask] = -1e6
                scores[b, obj] = tgt
                ranks = 1 + torch.argsort(
                    torch.argsort(scores, dim=1, descending=True),
                    dim=1
                )[b, obj]
                results['count'] += ranks.numel()
                results['mr']    += ranks.sum().item()
                results['mrr']   += (1.0/ranks.float()).sum().item()
                for k in [1,3,10]:
                    results[f'hits@{k}'] += (ranks<=k).sum().item()
        return results

    def evaluate(self, split):
        left  = self._predict(split, 'tail')
        right = self._predict(split, 'head')
        cnt = float(left['count'])
        out = {
            'left_mr': left['mr']/cnt,  'left_mrr': left['mrr']/cnt,
            'right_mr': right['mr']/cnt,'right_mrr': right['mrr']/cnt,
        }
        out['mr']  = 0.5*(out['left_mr']+out['right_mr'])
        out['mrr'] = 0.5*(out['left_mrr']+out['right_mrr'])
        for k in [1,3,10]:
            out[f'hits@{k}'] = 0.5*(left[f'hits@{k}']/cnt + right[f'hits@{k}']/cnt)
        return out

    def fit(self):
        os.makedirs('checkpoints', exist_ok=True)
        ckpt = f"checkpoints/{self.p.name}.pt"
        if self.p.restore and os.path.exists(ckpt):
            self.model.load_state_dict(torch.load(ckpt)['model'])
            logger.info("Restored from previous checkpoint.")

        # Early stopping settings
        patience = 10
        threshold = 0.0005
        no_improve_epochs = 0
        prev_mrr = None

        for epoch in range(1, self.p.max_epochs + 1):
            t0 = time.time()
            loss = self.train()
            val = self.evaluate('valid')
            curr_mrr = val['mrr']

            # Save best model
            if curr_mrr > self.best_val_mrr:
                self.best_val_mrr = curr_mrr
                self.best_val_results = val
                self.best_epoch = epoch
                torch.save({'model': self.model.state_dict()}, ckpt)
                logger.info(f"Epoch {epoch}: new best MRR={curr_mrr:.4f}, model saved.")
            else:
                logger.info(f"Epoch {epoch}: MRR={curr_mrr:.4f}, loss={loss:.4f}, time={time.time()-t0:.1f}s")

            # Check early stopping
            if prev_mrr is not None and abs(curr_mrr - prev_mrr) < threshold:
                no_improve_epochs += 1
            else:
                no_improve_epochs = 0
            prev_mrr = curr_mrr

            if no_improve_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no MRR change ≥{threshold} for {patience} epochs)")
                break

        # Test evaluation
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model'])
        test = self.evaluate('test')
        logger.info(
            f"Test results — MRR: {test['mrr']:.4f}, Hits@1: {test['hits@1']:.4f}, "
            f"Hits@3: {test['hits@3']:.4f}, Hits@10: {test['hits@10']:.4f}"
        )
        print(f"Test MRR={test['mrr']:.4f}, Hits@1={test['hits@1']:.4f}, Hits@3={test['hits@3']:.4f}, Hits@10={test['hits@10']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CompGCN on Knowledge Graph',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--name', default='test_run')
    parser.add_argument('--data', dest='dataset', default='FB15k-237')
    parser.add_argument('--score_func', dest='score_func', default='conve')
    parser.add_argument('--decoder_model', dest='decoder_model', choices=['conve','distmult'], default='conve')
    parser.add_argument('--opn', dest='opn', default='mult')
    parser.add_argument('--input_drop', dest='input_drop', type=float, default=0.2)
    parser.add_argument('--feat_drop', dest='feat_drop', type=float, default=0.2)
    parser.add_argument('--out_dim', dest='out_dim', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--num_bases', type=int, default=-1)
    parser.add_argument('--init_dim', type=int, default=100)
    parser.add_argument('--gcn_dim', type=int, default=400)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--gcn_drop', type=float, default=0.1)
    parser.add_argument('--hid_drop', type=float, default=0.3)
    parser.add_argument('--conve_hid_drop', type=float, default=0.3)
    parser.add_argument('--k_w', type=int, default=10)
    parser.add_argument('--k_h', type=int, default=10)
    parser.add_argument('--num_filt', type=int, default=200)
    parser.add_argument('--ker_sz', type=int, default=7)

    args = parser.parse_args()
    if args.decoder_model is None:
        args.decoder_model = args.score_func
    if args.out_dim is None:
        args.out_dim = args.embed_dim if args.embed_dim is not None else (args.k_w * args.k_h)

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu >=0 and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 日志配置
    LOG_FILE = 'log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=LOG_FILE,
        filemode='a'
    )
    logger = logging.getLogger()
    logger.info("========== Run Start ==========")
    logger.info(f"Parameters: {vars(args)}")

    Runner(args).fit()

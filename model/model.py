# coding: UTF-8
import re
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
import random
import torch.nn.functional as F
from .CGE import CGEConv
from .GAT import GraphAttentionLayer
from .Multi_GCN import GraphConvLayer
from .myGraph import my_CGEConv
import csv
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import ast
import networkx as nx
import json
import math
from torch_geometric.nn.conv import GraphConv
from torch.nn.utils import spectral_norm
embedding_size = 768
class focal_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += (1 - alpha)
            self.alpha[1:] += alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = F.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
class Classifier(nn.Module):
    def __init__(self, feat_dim=768, num_classes=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim, dtype=dtype))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, x):
        raise NotImplementedError

    def apply_weight(self, weight):
        self.weight.data = weight.clone()


class CosineClassifier(Classifier):
    def __init__(self, feat_dim=None, num_classes=None, dtype=None, scale=30, **kwargs):
        super().__init__(feat_dim, num_classes, dtype)
        self.scale = scale
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.scale
class bertCSRModel(nn.Module):
    def __init__(self, args):
        super(bertCSRModel, self).__init__()
        self.device = args.device
        self.pretrained_model = BertModel.from_pretrained(args.model_name)
        self.mlp = nn.Sequential(nn.Linear(1 * args.n_last+2 * embedding_size, args.mlp_size),#1 * args.n_last + 2 * embedding_size
                                  nn.ReLU(), nn.Dropout(args.mlp_drop),
                                  nn.Linear(args.mlp_size, args.no_of_classes))
        self.inter_mlp = nn.Sequential(nn.Linear(1 * args.n_last+2 * embedding_size, args.mlp_size),
                                       nn.ReLU(), nn.Dropout(args.mlp_drop),
                                       nn.Linear(args.mlp_size, args.no_of_classes))
        self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
        # self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        self.rate = args.rate
        self.w = args.w
        self.max_iteration = args.max_iteration + 1
        self.threshold = args.threshold
        self.min_iteration = args.min_iteration
        self.CGE = CGEConv(in_channels=args.in_channels,
                                 out_channels=args.n_last,
                                 metadata=args.metadata,
                                 heads=args.num_heads,
                                 beta_intra=args.beta_intra,
                                 beta_inter=args.beta_inter
                                 )  # 图编码
        self.inter_CGE = my_CGEConv(in_channels=args.in_channels,
                                 out_channels=args.n_last,
                                 metadata=args.metadata,
                                 heads=args.num_heads,
                                 beta_intra=args.beta_intra,
                                 beta_inter=args.beta_inter
                                 )  # 图编码
        self.GraphConv = GraphConvLayer(dropout=0.2,
                                        mem_dim=768,
                                        layers=2)
        self.linear = nn.Linear(3072, 2304)
        self.linear3 = nn.Linear(1536,768)
        self.sig = nn.Sigmoid()
        self.focal_loss = focal_loss(gamma=args.gamma, num_classes=args.no_of_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.loss_type = args.loss_type
    def forward(self, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs,all_senlable,all_senlable2,type,doc_id,tokenizer):
        sent_emb = self.pretrained_model(enc_input_ids,enc_mask_ids)[0] # 句子数，105,768
        yuan_sent_emb = sent_emb
        sent_embed = sent_emb.mean(1).unsqueeze(0)  # 句子数，768
        adj = self.GraphConv.build_graph(sent_embed)
        adj= F.softmax(adj, dim=-1)
        sent_embed = self.GraphConv(adj, sent_embed).squeeze(0)
        sent_embed = sent_embed.unsqueeze(1).repeat(1, 105, 1)  # 句子数，105，768
        sent_emb = sent_embed + sent_emb
        sent_list = []
        sent_ground = []
        for idx, event in enumerate(node_event):
            sent_id = int(event[0])
            sent_ground.append(sent_id)
            # sent_id = sent_ids[idx]
            event_pid = event[1]
            # event_pid = event_pids[idx]
            e_emb = self.extract_event(sent_emb[sent_id], event_pid).to(self.device)
            if idx == 0:
                event_embed = e_emb
            else:
                event_embed = torch.cat((event_embed, e_emb))
            if sent_id not in sent_list:
                sent_list.append(sent_id)
            sent_ground.append(sent_id)
        yuan_event_embed=event_embed
        if self.max_iteration > sent_list.__len__():
            max_iteration = sent_list.__len__()
        else:
            max_iteration = self.max_iteration
        pad_event1_pos = pad_sequence([torch.tensor(pos) for pos in t1_pos]).t().to(self.device).long()  # 3,435
        pad_event2_pos = pad_sequence([torch.tensor(pos) for pos in t2_pos]).t().to(self.device).long()
        event1 = torch.index_select(event_embed, 0, pad_event1_pos[0])
        event2 = torch.index_select(event_embed, 0, pad_event2_pos[0])  # 435,768
        event_pair_embed = torch.cat([event1, event2], dim=-1)  # 435,1536
        yuan_sent_emb=yuan_sent_emb.mean(1) #句子数，768
        for i,pair in enumerate(event_pairs):
            sent1=sent_ground[pair[0]]
            sent2=sent_ground[pair[1]]
            sent=torch.cat((yuan_sent_emb[sent1].unsqueeze(0),yuan_sent_emb[sent2].unsqueeze(0)),dim=0).unsqueeze(0)#1，2，768
            if i==0:
                true_sent = sent
            else:
                true_sent = torch.cat((true_sent,sent),dim=0) # 事件对数，2，768
        loss = 0.0
        iteration = 0
        difference = self.threshold

        if len(event_pairs) > 1:
            target = torch.cat([torch.tensor(t) for t in target], dim=0).to(self.device)
        else:
            target = torch.tensor(target[0]).to(self.device)

        # CGE
        event_embed = self.CGE.proj['event'](event_embed)
        ground_type = []
        for i, row in enumerate(rel_type[0]):
            if row==1:  # 句间
                ground_type.append(0)
            else:
                ground_type.append(1)
        intra_event_pairs = []
        inter_event_pairs = []
        intra_target = []
        inter_target = []
        intra_rel_type = [[]]
        inter_rel_type = [[]]
        intra_loss = 0.0
        inter_loss = 0.0
        intra_event_pair_embed = []
        inter_event_pair_embed = []
        inter_all_senlable=[]
        intra_all_senlable=[]
        intra_sent=[]
        inter_sent=[]
        # 分开
        for i, row in enumerate(ground_type):
            if ground_type[i] == 1:  # 句内
                intra_event_pairs.append(event_pairs[i])
                intra_target.append(target[i])
                intra_rel_type[0].append(rel_type[0][i])
                intra_event_pair_embed.append(event_pair_embed[i])
                intra_all_senlable.append(all_senlable[i])
                intra_sent.append(true_sent[i])
            else:
                inter_event_pairs.append(event_pairs[i])
                inter_target.append(target[i])
                inter_rel_type[0].append(rel_type[0][i])
                inter_event_pair_embed.append(event_pair_embed[i])
                inter_all_senlable.append(all_senlable2[i])
                inter_sent.append(true_sent[i])
        intra_target = torch.tensor(intra_target).to(self.device)
        inter_target = torch.tensor(inter_target).to(self.device)
        # 句内
        if len(intra_event_pairs) >= 1:
            intra_event_pair_embed = torch.stack(intra_event_pair_embed)
            intra_all_senlable = torch.stack(intra_all_senlable)
            intra_sent = torch.stack(intra_sent)
            intra_prediction, intra_loss = self.loop_prediction(iteration, max_iteration, difference, intra_event_pairs,
                                                                event_embed,intra_target,intra_rel_type, intra_loss,
                                                                intra_event_pair_embed,intra_all_senlable,yuan_sent_emb,yuan_event_embed,type='intra')

        else:
            intra_prediction = []
        # 句间
        if len(inter_event_pairs) >= 1:
            inter_event_pair_embed = torch.stack(inter_event_pair_embed)
            # inter_event_pair_long_embed = torch.stack(inter_event_pair_long_embed)
            inter_all_senlable = torch.stack(inter_all_senlable)
            inter_sent = torch.stack(inter_sent)
            inter_prediction, inter_loss = self.loop_prediction(iteration, max_iteration, difference, inter_event_pairs,
                                                                event_embed,inter_target,inter_rel_type, inter_loss,
                                                                inter_event_pair_embed,inter_all_senlable,yuan_sent_emb,yuan_event_embed,type='inter')
        else:
            inter_prediction = []
        # all_loss=0.0
        # all_prediction, all_loss = self.loop_prediction(iteration, max_iteration, difference, event_pairs,
        #                                                     event_embed, target, rel_type, all_loss,
        #                                                     event_pair_embed, all_senlable, yuan_sent_emb,sent_emb,type='all')
        x = 0
        c = 0
        prediction = []
        for i, row in enumerate(ground_type):
            if ground_type[i] == 1:
                prediction.append(intra_prediction[x])
                x = x + 1
            else:
                prediction.append(inter_prediction[c])
                c = c + 1
        prediction = torch.stack(prediction)
        loss += intra_loss * 0.5 + inter_loss * 0.5
        return loss, prediction

    def loop_prediction(self,iteration,max_iteration,difference,event_pairs,event_embed,target,rel_type,loss,event_pair_embed,all_senlable,sent_emb,yuan_event_embed,type):
        if type == 'intra':
            mlp = self.mlp
            CGE = self.CGE
            threshold = self.threshold
        else:
            mlp = self.inter_mlp
            CGE = self.inter_CGE
            threshold = self.threshold
        w = 6
        c=0
        ground_sent=None
        while iteration < max_iteration and (iteration < self.min_iteration or difference > threshold):
            if len(event_pairs) > 1:
                event_diff = torch.cat(
                    [event_embed[[pair[0]]] - event_embed[[pair[1]]] for pair in event_pairs])  # 55,768
            else:
                event_diff = event_embed[[event_pairs[0][0]]] - event_embed[[event_pairs[0][1]]]
            event_pair_pre = torch.cat([event_pair_embed, event_diff], dim=1)# 1536+768
            prediction = mlp(event_pair_pre)
            if self.loss_type == 'focal':
                loss += (1 / (iteration + 1)) * self.focal_loss(prediction, target)
            else:
                loss += (1 / (iteration + 1)) * self.ce_loss(prediction, target)
            # CGC
            graphedge_index = self.get_graphedge_index(prediction, event_pairs, rel_type)
            # CGE
            if graphedge_index.__len__() > 0:
                node_length = event_embed.size()[0]
                self_loop = torch.cat(
                    (torch.arange(node_length).unsqueeze(0), torch.arange(node_length).unsqueeze(0)),
                    dim=0).to(self.device)
                if ('event', 'intra', 'event') in graphedge_index:
                    graphedge_index[('event', 'intra', 'event')] = torch.cat(
                        (graphedge_index[('event', 'intra', 'event')], self_loop), dim=-1).long()
                else:
                    graphedge_index[('event', 'intra', 'event')] = self_loop.long()

                event_han = {'event': event_embed}
                event_han = CGE(event_han, graphedge_index)
                event_embed = event_han['event']
            else:
                event_embed = CGE.proj['event'](event_embed)

            if iteration > 0:
                difference = self.Contrast_pre(prediction, prediction_1)
            if len(event_pairs) > 1:
                event_pair = torch.cat([torch.cat([event_embed[[pair[0]]],event_embed[[pair[1]]]],dim=1) for pair in event_pairs])  # 55,768
            else:
                event_pair = torch.cat([event_embed[[event_pairs[0][0]]],event_embed[[event_pairs[0][1]]]],dim=1)
            event_pair = self.linear3(event_pair)
            event_pair = F.normalize(event_pair, p=2, dim=1)  # (实体对数, 768)
            sent_emb = F.normalize(sent_emb, p=2, dim=1)  # (句子数, 768)
            event_expanded = event_pair.unsqueeze(1).expand(-1, sent_emb.size(0),-1)  # [事件对数, 句子数, 768]
            sentence_expanded = sent_emb.squeeze(1).unsqueeze(0).expand(event_pair.size(0), -1,-1)  # [事件对数, 句子数, 768]
            cosine_sim=F.cosine_similarity(event_expanded,sentence_expanded,dim=2)# 事件对数，句子数
            # cosine_sim = F.softmax(cosine_sim,dim=1)
            flag=float(1/cosine_sim.size(1))
            cosine_sim=(cosine_sim>flag).float()
            sen_loss = self.bce_loss(cosine_sim.float(), all_senlable.float())
            if self.loss_type == 'focal':
                loss += (1 / (iteration + 1))* w *sen_loss
            else:
                loss += (1 / (iteration + 1)) * self.ce_loss(prediction, target)
            iteration += 1
            prediction_1 = prediction
            c=c+1
        return prediction,loss
    def extract_event(self, embed, event_pid):
        e_1 = int(event_pid[0])
        e_2 = int(event_pid[1])
        e1_embed = torch.zeros(1, embedding_size).to(self.device)
        length = e_2 - e_1
        for j in range(e_1,e_2):
            e1_embed += embed[j]
        event_embed = e1_embed / (length)
        return event_embed
    def build_index(self, event_emb, graphedge_index):
        num_nodes = event_emb.size(0)
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[graphedge_index[0], graphedge_index[1]] = 1
        return adj_matrix

    def get_graphedge_index(self, prediction, event_pair, rel_type):
        graphedge_index = {}
        if self.training:
            rate = self.rate
        else:
            rate = self.w
        if rate != 0:
            pred_soft = torch.softmax(prediction, dim=1)
            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (
                        torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and
                        rel_type[0][j] == 0)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (
                        torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and
                        rel_type[0][j] == 0)]).t().to(self.device)
            if tmp.size()[0] > 0:
                intra_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if (
                                                                  torch.max(pred_soft[j], dim=0)[0] > rate and
                                                                  torch.max(pred_soft[j], dim=0)[1] == 1 and
                                                                  rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                intra_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if (
                                                                  torch.max(pred_soft[j], dim=0)[0] > rate and
                                                                  torch.max(pred_soft[j], dim=0)[1] == 2 and
                                                                  rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_2 = tmp_1
            intra_graphedge_index = torch.cat((intra_graphedge_index_1, intra_graphedge_index_2), dim=-1)

            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (
                        torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and
                        rel_type[0][j] == 1)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (
                        torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and
                        rel_type[0][j] == 1)]).t().to(self.device)
            if tmp.size()[0] > 0:
                inter_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if (
                                                                  torch.max(pred_soft[j], dim=0)[0] > rate and
                                                                  torch.max(pred_soft[j], dim=0)[1] == 1 and
                                                                  rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                inter_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if (
                                                                  torch.max(pred_soft[j], dim=0)[0] > rate and
                                                                  torch.max(pred_soft[j], dim=0)[1] == 2 and
                                                                  rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_2 = tmp_1
            inter_graphedge_index = torch.cat((inter_graphedge_index_1, inter_graphedge_index_2), dim=-1)

        else:
            predt = torch.argmax(prediction, dim=1)
            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if
                                (predt[j] == 1 and rel_type[0][j] == 0)]).t().to('cuda')
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if
                                  (predt[j] == 2 and rel_type[0][j] == 0)]).t().to('cuda')
            if tmp.size()[0] > 0:
                intra_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if
                                                      (predt[j] == 1 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                intra_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if
                                                      (predt[j] == 2 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_2 = tmp_1
            intra_graphedge_index = torch.cat((intra_graphedge_index_1, intra_graphedge_index_2), dim=-1)

            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if
                                (predt[j] == 1 and rel_type[0][j] == 1)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if
                                  (predt[j] == 2 and rel_type[0][j] == 1)]).t().to(self.device)
            if tmp.size()[0] > 0:
                inter_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if
                                                      (predt[j] == 1 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                inter_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(
                    self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in
                                                      enumerate(event_pair) if
                                                      (predt[j] == 2 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_2 = tmp_1
            inter_graphedge_index = torch.cat((inter_graphedge_index_1, inter_graphedge_index_2), dim=-1)
        event_sent = []
        if intra_graphedge_index.size()[0] > 0:
            graphedge_index[('event', 'intra', 'event')] = intra_graphedge_index.long()
        if inter_graphedge_index.size()[0] > 0:
            graphedge_index[('event', 'inter', 'event')] = inter_graphedge_index.long()
        return graphedge_index
    def Contrast_pre(self, prediction, prediction_last):
        if self.training:
            rate = self.rate
        else:
            rate = self.w
        if rate != 0:
            pred_soft = torch.softmax(prediction, dim=1)
            pred_last_soft = torch.softmax(prediction_last, dim=1)
            pre_list = torch.tensor([]).to(self.device)
            pre_last_list = torch.tensor([]).to(self.device)
            max_pro, pred_t = torch.max(pred_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_list = torch.cat((pre_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_list = torch.cat((pre_list, torch.tensor([0]).to(self.device)), dim=-1)
            max_pro, pred_t = torch.max(pred_last_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_last_list = torch.cat((pre_last_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_last_list = torch.cat((pre_last_list, torch.tensor([0]).to(self.device)), dim=-1)
        else:
            pre_list = torch.argmax(prediction, dim=1)
            pre_last_list = torch.argmax(prediction_last, dim=1)
        different = (pre_list != pre_last_list).sum().item()
        return different

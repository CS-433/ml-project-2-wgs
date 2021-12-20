"""
@Project  : MLProject2Attention
@File     : AttentionAnalyzer.py
@Author   : Shaobo Cui
@Date     : 12/14/21 8:12 PM
"""
import json

import torch
import torch.nn as nn

from model.ema import EMA
import evaluate


class AttentionAnalyzer(object):
    def __init__(self, model, data):
        super(AttentionAnalyzer, self).__init__()
        self.model = model
        self.data = data

        self.attention_score_list = []
        self.device = None

    def attention_score(self, ema, args):
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

        j = 0

        def construct_mask(context_len, c_lens):
            # shape: [batch_size, context_len]
            context_mask = torch.arange(context_len).to(self.device).expand(len(c_lens), context_len) < c_lens.unsqueeze(1)
            return context_mask

        def summarize_attention_score(attention_scores, c_lens):
            # https://discuss.pytorch.org/t/variable-length-sequence-avg-pool-how-to-slice-based-off-a-long-tensor-of-slice-ends/37392/3
            # attention_scores.size(): [num_layers, batch_size, context_len]
            assert attention_scores.shape[1] == c_lens.shape[0]
            num_layers, batch_size, context_len = attention_scores.size()
            context_mask = construct_mask(context_len, c_lens)

            L_dup_context_mask = context_mask.unsqueeze(0).repeat(num_layers, 1, 1)
            masked_attentioni_scores = attention_scores * L_dup_context_mask

            # [num_layers, batch_size]
            sum_attention_scores = masked_attentioni_scores.sum(2)
            # it average over context_len: [num_layers, batch_size]
            average_attention_scores = sum_attention_scores / c_lens.unsqueeze(0).float()
            return average_attention_scores

        criterion = nn.CrossEntropyLoss()
        loss = 0
        answers = dict()
        self.model.eval()

        backup_params = EMA(0)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                backup_params.register(name, param.data)
                param.data.copy_(ema.get(name))

        with torch.set_grad_enabled(False):

            for batch in iter(self.data.dev_iter):
                p1, p2, attn_w, c_lens = self.model(batch)
                j += 1
                attention_scores = attn_w.permute((2, 0, 1))
                self.attention_score_list.append(summarize_attention_score(attention_scores, c_lens))
                print('Current batch no.: {}.'.format(j))

                batch_loss = criterion(p1, batch.s_idx) + criterion(p2, batch.e_idx)
                loss += batch_loss.item()

                # (batch, c_len, c_len)
                batch_size, c_len = p1.size()
                ls = nn.LogSoftmax(dim=1)
                mask = (torch.ones(c_len, c_len) * float('-inf')).to(self.device).tril(-1).unsqueeze(0).expand(
                    batch_size, -1, -1)
                score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
                score, s_idx = score.max(dim=1)
                score, e_idx = score.max(dim=1)
                s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

                for i in range(batch_size):
                    id = batch.id[i]
                    answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
                    answer = ' '.join([self.data.WORD.vocab.itos[idx] for idx in answer])
                    answers[id] = answer

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(backup_params.get(name))

        with open(args.prediction_file, 'w', encoding='utf-8') as f:
            print(json.dumps(answers), file=f)

        results = evaluate.main(args)
        print('EM: {} F1: {}'.format(results['exact_match'], results['f1']))

        self.attention_score_list = torch.cat(self.attention_score_list, dim=1)
        torch.save(self.attention_score_list, args.attn_tensor_save_path)



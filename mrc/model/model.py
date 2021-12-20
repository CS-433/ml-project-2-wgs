import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import LSTM, Linear
from model.lmkm import KMA


class BiDAF(nn.Module):
    def __init__(self, args, pretrained, lm_ffn_weights, lm_activate_layers):
        super(BiDAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Sequential(
            nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)),
            nn.ReLU()
            )

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(args.hidden_size * 2, 1)
        self.att_weight_q = Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = Linear(args.hidden_size * 2, 1)

        # PLUS Pretrained LM FFN Attention
        if self.args.add_lm_km in ['modeling', 'output', 'random']:
            self.lm_km_attn = KMA(lm_ffn_weights=lm_ffn_weights,
                                  activate_layers=lm_activate_layers,
                                  hidden_size=args.hidden_size,
                                  activation_memory_coef=self.args.lm_km_act_mc,
                                  add_bias=self.args.lm_km_add_bias,
                                  dropout=self.args.dropout,
                                  q_map=self.args.lm_km_q_map,
                                  random_weights=(self.args.add_lm_km == 'random'))
            self.km_size = self.lm_km_attn.get_ffn_hk_size()
        else:
            self.lm_km_attn = None
            self.km_size = None

        # 5. Modeling Layer
        '''
        if self.args.add_lm_km == 'modeling':
            self.modeling_LSTM1 = LSTM(input_size=args.hidden_size * 4 + self.km_size,
                                       hidden_size=args.hidden_size,
                                       bidirectional=True,
                                       batch_first=True,
                                       dropout=args.dropout)
        else:
            self.modeling_LSTM1 = LSTM(input_size=args.hidden_size * 4,
                                       hidden_size=args.hidden_size,
                                       bidirectional=True,
                                       batch_first=True,
                                       dropout=args.dropout)
        '''
        '''
        self.modeling_LSTM2 = LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)
        '''

        # 6. Output Layer
        if self.args.add_lm_km in ['modeling', 'random']:
            # self.p1_weight_g = Linear(args.hidden_size * 4 + self.km_size, 1, dropout=args.dropout)
            self.p1_weight_m = Linear(args.hidden_size * 4 + self.km_size, 1, dropout=args.dropout)
            # self.p2_weight_g = Linear(args.hidden_size * 4 + self.km_size, 1, dropout=args.dropout)
            self.p2_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
            self.output_LSTM = LSTM(input_size=args.hidden_size * 4 + self.km_size,
                                    hidden_size=args.hidden_size,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=args.dropout)
            self.output_LSTM_k = None
        elif self.args.add_lm_km == 'output':
            # self.p1_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
            self.p1_weight_m = Linear(args.hidden_size * 4, 1, dropout=args.dropout)
            self.p1_weight_k = Linear(self.km_size, 1, dropout=args.dropout)
            # self.p2_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
            self.p2_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
            self.p2_weight_k = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
            self.output_LSTM = LSTM(input_size=args.hidden_size * 4,
                                    hidden_size=args.hidden_size,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=args.dropout)
            self.output_LSTM_k = LSTM(input_size=self.km_size,
                                      hidden_size=args.hidden_size,
                                      bidirectional=True,
                                      batch_first=True,
                                      dropout=args.dropout)
        else:
            # self.p1_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
            self.p1_weight_m = Linear(args.hidden_size * 4, 1, dropout=args.dropout)
            # self.p2_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
            self.p2_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
            self.output_LSTM = LSTM(input_size=args.hidden_size * 4,
                                    hidden_size=args.hidden_size,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=args.dropout)
            self.output_LSTM_k = None

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch， seq_len, char_dim, word_len)
            x = x.transpose(2, 3)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(3)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # cq_tiled = c_tiled * q_tiled
            # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            # b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            # q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            # q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 4)
            x = torch.cat([c, c2q_att], dim=-1)
            return x

        def output_layer_a(m, l):
            """
            :param g: (batch, c_len, hidden_size * 8) or (batch, c_len, hidden_size * 8 + km_size)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            # p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            p1 = self.p1_weight_m(m).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            # p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()
            p2 = self.p2_weight_m(m2).squeeze()

            return p1, p2

        def output_layer_b(m, k, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :param k: (batch, c_len ,km_size)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            # p1 = (self.p1_weight_g(g) + self.p1_weight_m(m) + self.p1_weight_k(k)).squeeze()
            p1 = (self.p1_weight_m(m) + self.p1_weight_k(k)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            k2 = self.output_LSTM_k((k, l))[0]
            # (batch, c_len)
            # p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2) + self.p2_weight_k(k)).squeeze()
            p2 = (self.p2_weight_m(m2) + self.p2_weight_k(k2)).squeeze()

            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)

        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)

        # 3. Contextual Embedding Layer
        # (batch, c_len, hidden_size * 2)
        c = self.context_LSTM((c, c_lens))[0]
        # (batch, q_len, hidden_size * 2)
        q = self.context_LSTM((q, q_lens))[0]

        # 4. Attention Flow Layer
        # (batch, c_len, hidden_size * 4)
        g = att_flow_layer(c, q)

        # PLUS: Pretrained LM FFN Attention
        k = None
        attn_w = None
        # (batch, c_len, h_km), (batch, c_len, L)
        if self.args.add_lm_km in ['modeling', 'output', 'random']:
            k, attn_w = self.lm_km_attn.forward(g)

        # 5. Modeling Layer
        # (batch, c_len, hidden_size * 4 + h_km)
        if self.args.add_lm_km in ['modeling', 'random']:
            m = torch.cat([g, k], 2)
        else:
            m = g
        # m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # (batch, c_len, hidden_size * 2)
        # m = self.modeling_LSTM1((g, c_lens))[0]

        # 6. Output Layer
        if self.args.add_lm_km == 'output':
            p1, p2 = output_layer_b(m, k, c_lens)
        else:
            p1, p2 = output_layer_a(m, c_lens)

        # (batch, c_len), (batch, c_len), (batch, c_len, L), (batch)
        return p1, p2, attn_w, c_lens

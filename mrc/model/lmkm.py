import torch
from torch import nn
import torch.nn.functional as function
from transformers import AutoConfig, AutoTokenizer, AutoModel
from utils.nn import Linear


class PLMKM(object):
    def __init__(self, name='roberta-base', lm_type='encoder'):
        self.name = name
        self.lm_type = lm_type
        self.config = AutoConfig.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModel.from_pretrained(name, config=self.config)
        self.lm_state_dict = self.lm.state_dict().copy()

        if "bert" in self.name:  # for bert and roberta
            self.encoder_layer_num = self.lm.encoder.config.num_hidden_layers
            self.encoder_layer_state = {"layer": "encoder.layer.", "key": ".intermediate.dense",
                                        "value": ".output.dense"}
            self.decoder_layer_num = None
            self.decoder_layer_state = None
        elif "gpt2" in self.name:
            self.encoder_layer_num = None
            self.encoder_layer_state = None
            self.decoder_layer_num = self.lm.config.num_hidden_layers
            self.decoder_layer_state = {"layer": "h.", "key": ".mlp.c_fc", "value": ".mlp.c_proj"}
        elif "bart" in self.name and self.lm_type == "encoder":
            self.encoder_layer_num = self.lm.encoder.config.num_hidden_layers
            self.encoder_layer_state = {"layer": "encoder.layers.", "key": ".fc1", "value": ".fc2"}
            self.decoder_layer_num = None
            self.decoder_layer_state = None
        elif "bart" in self.name and self.lm_type == "decoder":
            self.encoder_layer_num = None
            self.encoder_layer_state = None
            self.decoder_layer_num = self.lm.decoder.config.num_hidden_layers
            self.decoder_layer_state = {"layer": "decoder.layers.", "key": ".fc1", "value": ".fc2"}
        elif "bart" in self.name and self.lm_type == "both":
            self.encoder_layer_num = self.lm.encoder.config.num_hidden_layers
            self.encoder_layer_state = {"layer": "encoder.layers.", "key": ".fc1", "value": ".fc2"}
            self.decoder_layer_num = self.lm.decoder.config.num_hidden_layers
            self.decoder_layer_state = {"layer": "decoder.layers.", "key": ".fc1", "value": ".fc2"}
        '''
        elif "xlnet" in self.name:
            self.encoder_layer_num = self.lm.encoder.n_layer
            self.encoder_layer_state = {"layer": "layer.", "key": ".ff.layer_1", "value": ".ff.layer_2"}
            self.decoder_layer_num = None
            self.decoder_layer_state = None
        '''

        if self.encoder_layer_state is not None:
            self.encoder_ffn_weights = {}
            layer_str = self.encoder_layer_state["layer"]
            key_str = self.encoder_layer_state["key"]
            value_str = self.encoder_layer_state["value"]
            for i in range(self.encoder_layer_num):
                # Tensor(3072, 768)
                self.encoder_ffn_weights["layer_"+str(i)+"_key_weight"] = \
                    self.lm_state_dict[layer_str + str(i) + key_str + ".weight"]
                # Tensor(3072)
                self.encoder_ffn_weights["layer_" + str(i) + "_key_bias"] = \
                    self.lm_state_dict[layer_str + str(i) + key_str + ".bias"]
                # Tensor(768, 3072)
                self.encoder_ffn_weights["layer_" + str(i) + "_value_weight"] = \
                    self.lm_state_dict[layer_str + str(i) + value_str + ".weight"]
                # Tensor(768)
                self.encoder_ffn_weights["layer_" + str(i) + "_value_bias"] = \
                    self.lm_state_dict[layer_str + str(i) + value_str + ".bias"]
        else:
            self.encoder_ffn_weights = None

        if self.decoder_layer_state is not None:
            self.decoder_ffn_weights = {}
            layer_str = self.decoder_layer_state["layer"]
            key_str = self.decoder_layer_state["key"]
            value_str = self.decoder_layer_state["value"]
            for i in range(self.decoder_layer_num):
                # Tensor(3072, 768)
                if "gpt2" in self.name:
                    self.decoder_ffn_weights["layer_" + str(i) + "_key_weight"] = \
                        self.lm_state_dict[layer_str + str(i) + key_str + ".weight"].transpose(1, 0).contiguous()
                else:
                    self.decoder_ffn_weights["layer_" + str(i) + "_key_weight"] = \
                        self.lm_state_dict[layer_str + str(i) + key_str + ".weight"]
                # Tensor(3072)
                self.decoder_ffn_weights["layer_" + str(i) + "_key_bias"] = \
                    self.lm_state_dict[layer_str + str(i) + key_str + ".bias"]
                # Tensor(768, 3072)
                if "gpt2" in self.name:
                    self.decoder_ffn_weights["layer_" + str(i) + "_value_weight"] = \
                        self.lm_state_dict[layer_str + str(i) + value_str + ".weight"].transpose(1, 0).contiguous()
                else:
                    self.decoder_ffn_weights["layer_" + str(i) + "_value_weight"] = \
                        self.lm_state_dict[layer_str + str(i) + value_str + ".weight"]
                # Tensor(768)
                self.decoder_ffn_weights["layer_" + str(i) + "_value_bias"] = \
                    self.lm_state_dict[layer_str + str(i) + value_str + ".bias"]
        else:
            self.decoder_ffn_weights = None

    def get_ffn_weights(self):
        return {"encoder_layer_num": self.encoder_layer_num, "encoder_ffn_weights": self.encoder_ffn_weights,
                "decoder_layer_num": self.decoder_layer_num, "decoder_ffn_weights": self.decoder_ffn_weights}

    def print_lm_config(self):
        print("Model: " + self.name)
        print("Config:")
        print(self.config.get_config_dict(self.name))

    def print_ffn_size(self):
        if self.encoder_layer_num is not None:
            print("Encoder Layer Number:")
            print(self.encoder_layer_num)
            print("Encoder Feed Forward Layer (Key) Weight Size:")
            print(self.encoder_ffn_weights["layer_0_key_weight"].size())
            print("Encoder Feed Forward Layer (Value) Weight Size:")
            print(self.encoder_ffn_weights["layer_0_value_weight"].size())

        if self.decoder_layer_num is not None:
            print("Decoder Layer Number:")
            print(self.decoder_layer_num)
            print("Decoder Feed Forward Layer (Key) Weight Size:")
            print(self.decoder_ffn_weights["layer_0_key_weight"].size())
            print("Decoder Feed Forward Layer (Value) Weight Size:")
            print(self.decoder_ffn_weights["layer_0_value_weight"].size())


class QMLP(nn.Module):
    def __init__(self, emb_hidden_size, q_hidden_size, dropout=0.0):
        super().__init__()
        self.input_layer = Linear(emb_hidden_size, q_hidden_size, dropout)
        self.active_function = nn.Sigmoid()
        self.output_layer = Linear(q_hidden_size, q_hidden_size, dropout)

    def forward(self, x):
        return self.output_layer(self.active_function(self.input_layer(x)))

    def __call__(self, x):
        return self.forward(x)


class KMA(nn.Module):
    def __init__(self, lm_ffn_weights, activate_layers, hidden_size, dropout, activation_memory_coef='relu',
                 add_bias=True, q_map='linear', random_weights=False):
        super().__init__()
        self.hidden_size = hidden_size  # Hm
        self.act_memo_coef = activation_memory_coef
        self.add_bias = add_bias
        self.dropout = dropout
        self.q_map = q_map
        self.random_w = random_weights

        ffn_key_weights = []
        ffn_value_weights = []
        ffn_key_bias = []
        ffn_value_bias = []
        if activate_layers["encoder"] is not None:
            for idx in activate_layers["encoder"]:
                ffn_key_weights.append(lm_ffn_weights["encoder_ffn_weights"]["layer_" + str(idx) + "_key_weight"]
                                       .unsqueeze(0))
                ffn_value_weights.append(lm_ffn_weights["encoder_ffn_weights"]["layer_" + str(idx) + "_value_weight"]
                                         .unsqueeze(0))
                ffn_key_bias.append(lm_ffn_weights["encoder_ffn_weights"]["layer_" + str(idx) + "_key_bias"]
                                    .unsqueeze(0).unsqueeze(2))
                ffn_value_bias.append(lm_ffn_weights["encoder_ffn_weights"]["layer_" + str(idx) + "_value_bias"]
                                      .unsqueeze(0).unsqueeze(2))
        if activate_layers["decoder"] is not None:
            for idx in activate_layers["decoder"]:
                ffn_key_weights.append(lm_ffn_weights["decoder_ffn_weights"]["layer_" + str(idx) + "_key_weight"]
                                       .unsqueeze(0))
                ffn_value_weights.append(lm_ffn_weights["decoder_ffn_weights"]["layer_" + str(idx) + "_value_weight"]
                                         .unsqueeze(0))
                ffn_key_bias.append(lm_ffn_weights["decoder_ffn_weights"]["layer_" + str(idx) + "_key_bias"]
                                    .unsqueeze(0).unsqueeze(2))
                ffn_value_bias.append(lm_ffn_weights["decoder_ffn_weights"]["layer_" + str(idx) + "_value_bias"]
                                      .unsqueeze(0).unsqueeze(2))
        # L * [1, Inter, Hk] -> [L, Inter, Hk], not update
        ffn_key_weights = torch.cat(ffn_key_weights, 0)
        # L * [1, Hk, Inter] -> [L, Hk, Inter], not update
        ffn_value_weights = torch.cat(ffn_value_weights, 0)
        # L * [1, Inter, 1] -> [L, Inter, 1], not update
        ffn_key_bias = torch.cat(ffn_key_bias, 0)
        # L * [1, Hk, 1] -> [L, Hk, 1], not update
        ffn_value_bias = torch.cat(ffn_value_bias, 0)

        self.ffn_layer_num = int(ffn_key_weights.size(0))  # L
        self.ffn_inter_size = int(ffn_key_weights.size(1))  # or ffn_value_weights.size(2), Inter
        self.ffn_hk_size = int(ffn_key_weights.size(2))  # or ffn_value_weights.size(1), Hk

        if self.random_w:
            torch.random.manual_seed(0)
            self.register_buffer("ffn_key_weights", torch.randn(self.ffn_layer_num, self.ffn_inter_size, self.ffn_hk_size))
            self.register_buffer("ffn_value_weights", torch.randn(self.ffn_layer_num, self.ffn_hk_size, self.ffn_inter_size))
            self.register_buffer("ffn_key_bias", torch.randn(self.ffn_layer_num, self.ffn_inter_size, 1))
            self.register_buffer("ffn_value_bias", torch.randn(self.ffn_layer_num, self.ffn_hk_size, 1))
        else:
            self.register_buffer("ffn_key_weights", ffn_key_weights)
            self.register_buffer("ffn_value_weights", ffn_value_weights)
            self.register_buffer("ffn_key_bias", ffn_key_bias)
            self.register_buffer("ffn_value_bias", ffn_value_bias)

        if q_map == "mlp":
            self.embed_query_proj_inner_layers = \
                nn.ModuleList([QMLP(4 * self.hidden_size, self.ffn_hk_size, self.dropout) for _ in range(self.ffn_layer_num)])
            self.embed_query_proj_inter_layer = QMLP(4 * self.hidden_size, self.ffn_hk_size, self.dropout)
        else:
            self.embed_query_proj_inner_layers = \
                nn.ModuleList([Linear(4 * self.hidden_size, self.ffn_hk_size, self.dropout) for _ in range(self.ffn_layer_num)])
            self.embed_query_proj_inter_layer = Linear(4 * self.hidden_size, self.ffn_hk_size, self.dropout)

    def get_ffn_hk_size(self):
        return self.ffn_hk_size

    def forward(self, embeds):  # embeds: [B,T,4Hm] -> output_km: [B,T,Hk]

        # [B,T,4Hm] -> L * [B,1,T,Hk] -> [B,L,T,Hk]
        query_inner_layers = []
        for i in range(self.ffn_layer_num):
            query_inner_layers.append(self.embed_query_proj_inner_layers[i](embeds).unsqueeze(1))
        query_inner_layers = torch.cat(query_inner_layers, 1)

        # [L,Inter,Hk] -> [L,Hk,Inter]
        key_ffn_weights = self.ffn_key_weights.transpose(2, 1).contiguous()
        # [B,L,T,Hk] * [L,Hk,Inter] -> [B,L,T,Inter]
        energy_inner_layer = torch.matmul(query_inner_layers, key_ffn_weights)
        if self.add_bias:
            # [L,Inter,1] -> [L,1,Inter] -> [1,L,1,Inter]
            key_ffn_bias = self.ffn_key_bias.transpose(2, 1).unsqueeze(0).contiguous()
            energy_inner_layer = energy_inner_layer + key_ffn_bias

        # [B,L,T,Inter]
        if self.act_memo_coef == "relu":
            attn_weights_inner_layer = function.relu(energy_inner_layer)
        elif self.act_memo_coef == "softmax":
            attn_weights_inner_layer = function.softmax(energy_inner_layer, dim=3)
        else:
            attn_weights_inner_layer = energy_inner_layer

        # [L,Hk,Inter] -> [L,Inter,Hk]
        value_ffn_weights = self.ffn_value_weights.transpose(2, 1).contiguous()
        # [B,L,T,Inter] * [L,Inter,Hk] -> [B,L,T,Hk]
        output_inner_layer = torch.matmul(attn_weights_inner_layer, value_ffn_weights)
        if self.add_bias:
            # [L,Hk,1] -> [L,1,Hk] -> [1,L,1,Hk]
            value_ffn_bias = self.ffn_value_bias.transpose(2, 1).unsqueeze(0).contiguous()
            output_inner_layer = output_inner_layer + value_ffn_bias
        # [B,L,T,Hk] -> [B,T,L,Hk]
        output_inner_layer = output_inner_layer.transpose(2, 1).contiguous()

        # [B,T,4Hm] -> [B,T,Hk]
        query_inter_layer = self.embed_query_proj_inter_layer(embeds)
        # [B,T,Hk] -> [B,T,Hk,1]
        query_inter_layer = query_inter_layer.unsqueeze(3)
        # [B,T,L,Hk] * [B,T,Hk,1] -> [B,T,L,1]
        energy_inter_layer = torch.matmul(output_inner_layer, query_inter_layer)

        # [B,T,L,1]
        attn_weights_inter_layer = function.softmax(energy_inter_layer, dim=2)

        # [B,T,L,Hk] -> [B,T,Hk,L]
        output_inner_layer = output_inner_layer.transpose(3, 2).contiguous()
        # [B,T,Hk,L] * [B,T,L,1] -> [B,T,Hk,1]
        output_inter_layer = torch.matmul(output_inner_layer, attn_weights_inter_layer)
        # [B,T,Hk,1] -> [B,T,Hk]
        output_inter_layer = output_inter_layer.squeeze(3)

        # add Tanh activation, [B,T,Hk]
        output_km = torch.tanh(output_inter_layer)

        # return_weights = attn_weights_inter_layer.squeeze(3)
        return_weights = energy_inter_layer.squeeze(3)

        return output_km, return_weights

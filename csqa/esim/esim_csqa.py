# TODO: projection dropout with ELMO
#   l2 reg with ELMO
#   multiple ELMO layers
#   doc

from typing import Dict, Optional

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as function
from transformers import AutoConfig, AutoTokenizer, AutoModel

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, last_dim_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import CategoricalAccuracy

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x

    def __call__(self, x):
        return self.forward(x)


class QMLP(nn.Module):
    def __init__(self, emb_hidden_size, q_hidden_size, dropout=0.0):
        super().__init__()
        self.input_layer = Linear(emb_hidden_size, q_hidden_size, dropout)
        self.active_function = nn.ReLU()
        self.output_layer = Linear(q_hidden_size, q_hidden_size, dropout)

    def forward(self, x):
        return self.output_layer(self.active_function(self.input_layer(x)))

    def __call__(self, x):
        return self.forward(x)


class PLMKM(object):
    def __init__(self, name='bert-base-uncased', lm_type='encoder'):
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
                nn.ModuleList([QMLP(self.hidden_size, self.ffn_hk_size, self.dropout) for _ in range(self.ffn_layer_num)])
            self.embed_query_proj_inter_layer = QMLP(self.hidden_size, self.ffn_hk_size, self.dropout)
        else:
            self.embed_query_proj_inner_layers = \
                nn.ModuleList([Linear(self.hidden_size, self.ffn_hk_size, self.dropout) for _ in range(self.ffn_layer_num)])
            self.embed_query_proj_inter_layer = Linear(self.hidden_size, self.ffn_hk_size, self.dropout)

    def get_ffn_hk_size(self):
        return self.ffn_hk_size

    def forward(self, embeds):  # embeds: [B,T,Hm] -> output_km: [B,T,Hk]

        # [B,T,Hm] -> L * [B,T,Hk]
        query_inner_layers = []
        for i in range(self.ffn_layer_num):
            query_inner_layers.append(self.embed_query_proj_inner_layers[i](embeds))

        output_inner_layer_all = []
        for layer in range(self.ffn_layer_num):
            # [L,Inter,Hk] -> [Inter,Hk] -> [1,Hk,Inter]
            key_ffn_weight_layer = self.ffn_key_weights[layer, :, :].unsqueeze(0).transpose(2, 1).contiguous()
            # [B,T,Hk]
            query_inner_layer = query_inner_layers[layer]
            # [B,T,Hk] matmul [1,Hk,Inter] -> [B,T,Inter]
            energy_inner_layer = torch.matmul(query_inner_layer, key_ffn_weight_layer)
            if self.add_bias:
                # [L,Inter,1] -> [1,Inter,1] -> [1,1,Inter]
                key_ffn_bias = self.ffn_key_bias[layer, :, :].unsqueeze(0).transpose(2, 1).contiguous()
                energy_inner_layer = energy_inner_layer + key_ffn_bias

            # [B,T,Inter]
            if self.act_memo_coef == "relu":
                attn_weights_inner_layer = function.relu(energy_inner_layer)
            elif self.act_memo_coef == "softmax":
                attn_weights_inner_layer = function.softmax(energy_inner_layer, dim=3)
            else:
                attn_weights_inner_layer = energy_inner_layer

            # [L,Hk,Inter] -> [1,Hk,Inter] -> [1,Inter,Hk]
            value_ffn_weights = self.ffn_value_weights[layer, :, :].unsqueeze(0).transpose(2, 1).contiguous()
            # [B,T,Inter] matmul [1,Inter,Hk] -> [B,T,Hk]
            output_inner_layer = torch.matmul(attn_weights_inner_layer, value_ffn_weights)
            if self.add_bias:
                # [L,Hk,1] -> [1,Hk,1] -> [1,1,Hk]
                value_ffn_bias = self.ffn_value_bias[layer, :, :].unsqueeze(0).transpose(2, 1).contiguous()
                output_inner_layer = output_inner_layer + value_ffn_bias

            # L * [B,T,Hk]
            output_inner_layer_all.append(output_inner_layer)

        # [B,T,8Hm] -> [B,T,Hk]
        query_inter_layer = self.embed_query_proj_inter_layer(embeds)

        energy_inter_layer_all = []
        for layer in range(self.ffn_layer_num):
            # [B,T,Hk] * [B,T,Hk] -> [B,T,1]
            energy_inter_layer = torch.sum(output_inner_layer_all[layer] * query_inter_layer, dim=2).unsqueeze(2)
            # L * [B,T,1]
            energy_inter_layer_all.append(energy_inter_layer)
        # [B,T,L]
        energy_inter_layer_all = torch.cat(energy_inter_layer_all, 2)

        # [B,T,L]
        attn_weights_inter_layer = function.softmax(energy_inter_layer_all, dim=2)

        # [B,T,Hk]
        output_inter_layer = torch.zeros_like(output_inner_layer_all[0])
        for layer in range(self.ffn_layer_num):
            output_inter_layer += output_inner_layer_all[layer] * attn_weights_inter_layer[:, :, layer].unsqueeze(2)

        # add Tanh activation
        output_km = torch.tanh(output_inter_layer)

        return output_km


class VariationalDropout(nn.Dropout):
    def forward(self, input):
        """
        input is shape (batch_size, timesteps, embedding_dim)
        Samples one mask of size (batch_size, embedding_dim) and applies it to every time step.
        """
        # ones = Variable(torch.ones(input.shape[0], input.shape[-1]))
        ones = Variable(input.data.new(input.shape[0], input.shape[-1]).fill_(1))
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input


@Model.register("esim_csqa")
class ESIM(Model):
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    attend_feedforward : ``FeedForward``
        This feedforward network is applied to the encoded sentence representations before the
        similarity matrix is computed between words in the premise and words in the hypothesis.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between words in
        the premise and words in the hypothesis.
    compare_feedforward : ``FeedForward``
        This feedforward network is applied to the aligned premise and hypothesis representations,
        individually.
    aggregate_feedforward : ``FeedForward``
        This final feedforward network is applied to the concatenated, summed result of the
        ``compare_feedforward`` network, and its output is used as the entailment class logits.
    premise_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the premise, we can optionally apply an encoder.  If this is ``None``, we
        will do nothing.
    hypothesis_encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        After embedding the hypothesis, we can optionally apply an encoder.  If this is ``None``,
        we will use the ``premise_encoder`` for the encoding (doing nothing if ``premise_encoder``
        is also ``None``).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 # inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 dropout: float = 0.3,
                 hidden_size: int = 300,
                 add_lm_km: str = 'none',  # [modeling, none, random]
                 lm_km_name: str = 'bert-base-uncased',  # [bert-base-uncased, roberta-base, gpt2, facebook/bart-base]
                 lm_km_type: str = 'encoder',  # [encoder, decoder, both]
                 lm_km_act_mc: str = 'relu',  # [relu, softmax]
                 lm_km_add_bias: bool = True,
                 lm_km_q_map: str = 'linear',  # [linear, mlp]
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        print('loading pretrained language model...')
        lm_km = PLMKM(lm_km_name, lm_km_type)
        lm_km.print_lm_config()
        lm_km.print_ffn_size()
        ffn_weights = lm_km.get_ffn_weights()
        encoder_active_layer = None
        if ffn_weights["encoder_layer_num"] is not None:
            encoder_active_layer = [x for x in range(ffn_weights["encoder_layer_num"])]  # all layers
        decoder_active_layer = None
        if ffn_weights["decoder_layer_num"] is not None:
            decoder_active_layer = [x for x in range(ffn_weights["decoder_layer_num"])]  # all layers
        active_layers = {"encoder": encoder_active_layer, "decoder": decoder_active_layer}

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        self._matrix_attention = MatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        # PLUS Pretrained LM FFN Attention
        self.add_lm_km = add_lm_km
        if self.add_lm_km in ['modeling', 'random']:
            self._lm_km_attn = KMA(lm_ffn_weights=ffn_weights,
                                   activate_layers=active_layers,
                                   hidden_size=hidden_size,
                                   activation_memory_coef=lm_km_act_mc,
                                   add_bias=lm_km_add_bias,
                                   dropout=dropout,
                                   q_map=lm_km_q_map,
                                   random_weights=(self.add_lm_km == 'random'))
        else:
            self._lm_km_attn = None

        # self._inference_encoder = inference_encoder
        '''
        "inference_encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "num_layers": 1,
        "bidirectional": true
        },
        '''

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = VariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        # check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
        #                        "text field embedding dim", "encoder input dim")
        # check_dimensions_match(encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
        #                        "encoder output dim", "projection feedforward input")
        # check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
        #                        "proj feedforward output dim", "inference lstm input dim")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis0: Dict[str, torch.LongTensor],
                hypothesis1: Dict[str, torch.LongTensor],
                hypothesis2: Dict[str, torch.LongTensor],
                hypothesis3: Dict[str, torch.LongTensor],
                hypothesis4: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        hyps = [hypothesis0, hypothesis1, hypothesis2, hypothesis3, hypothesis4]
        if isinstance(self._text_field_embedder, ElmoTokenEmbedder):
            self._text_field_embedder._elmo._elmo_lstm._elmo_lstm.reset_states()

        embedded_premise = self._text_field_embedder(premise)

        embedded_hypotheses = []
        for hypothesis in hyps:
            if isinstance(self._text_field_embedder, ElmoTokenEmbedder):
                self._text_field_embedder._elmo._elmo_lstm._elmo_lstm.reset_states()
            embedded_hypotheses.append(self._text_field_embedder(hypothesis))

        premise_mask = get_text_field_mask(premise).float()
        hypothesis_masks = [get_text_field_mask(hypothesis).float() for hypothesis in hyps]
        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_premise = self.rnn_input_dropout(embedded_premise)
            embedded_hypotheses = [self.rnn_input_dropout(hyp) for hyp in embedded_hypotheses]

        # encode premise and hypothesis
        encoded_premise = self._encoder(embedded_premise, premise_mask)

        label_logits = []
        for i, (embedded_hypothesis, hypothesis_mask) in enumerate(zip(embedded_hypotheses, hypothesis_masks)):
            encoded_hypothesis = self._encoder(embedded_hypothesis, hypothesis_mask)

            # Shape: (batch_size, premise_length, hypothesis_length)
            similarity_matrix = self._matrix_attention(encoded_premise, encoded_hypothesis)

            # Shape: (batch_size, premise_length, hypothesis_length)
            p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
            # Shape: (batch_size, premise_length, embedding_dim)
            attended_hypothesis = weighted_sum(encoded_hypothesis, p2h_attention)

            # Shape: (batch_size, hypothesis_length, premise_length)
            h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
            # Shape: (batch_size, hypothesis_length, embedding_dim)
            attended_premise = weighted_sum(encoded_premise, h2p_attention)

            # the "enhancement" layer
            '''
            premise_enhanced = torch.cat(
                [encoded_premise, attended_hypothesis,
                 encoded_premise - attended_hypothesis,
                 encoded_premise * attended_hypothesis],
                dim=-1
            )
            hypothesis_enhanced = torch.cat(
                [encoded_hypothesis, attended_premise,
                 encoded_hypothesis - attended_premise,
                 encoded_hypothesis * attended_premise],
                dim=-1
            )
            '''
            premise_enhanced = torch.cat([encoded_premise, attended_hypothesis], dim=-1)
            hypothesis_enhanced = torch.cat([encoded_hypothesis, attended_premise], dim=-1)

            # embedding -> lstm w/ do -> enhanced attention -> dropout_proj, only if ELMO -> ff proj -> lstm w/ do -> dropout -> ff 300 -> dropout -> output

            # add dropout here with ELMO

            # the projection layer down to the model dimension
            # no dropout in projection
            projected_enhanced_premise = self._projection_feedforward(premise_enhanced)
            projected_enhanced_hypothesis = self._projection_feedforward(hypothesis_enhanced)

            knowledge_premise = None
            knowledge_hypothesis = None
            if self.add_lm_km in ['modeling', 'random']:
                knowledge_premise = self._lm_km_attn.forward(projected_enhanced_premise)
                knowledge_hypothesis = self._lm_km_attn.forward(projected_enhanced_hypothesis)

            if self.add_lm_km in ['modeling', 'random']:
                v_ai = torch.cat([projected_enhanced_premise, knowledge_premise], dim=-1)
                v_bi = torch.cat([projected_enhanced_hypothesis, knowledge_hypothesis], dim=-1)
            else:
                v_ai = projected_enhanced_premise
                v_bi = projected_enhanced_hypothesis

            # Run the inference layer
            '''
            if self.rnn_input_dropout:
                projected_enhanced_premise = self.rnn_input_dropout(projected_enhanced_premise)
                projected_enhanced_hypothesis = self.rnn_input_dropout(projected_enhanced_hypothesis)
            v_ai = self._inference_encoder(projected_enhanced_premise, premise_mask)
            v_bi = self._inference_encoder(projected_enhanced_hypothesis, hypothesis_mask)
            '''

            # The pooling layer -- max and avg pooling.
            # (batch_size, T, model_dim + k_dim) -> (batch_size, model_dim + k_dim)
            v_a_max, _ = replace_masked_values(v_ai, premise_mask.unsqueeze(-1), -1e7).max(dim=1)
            v_b_max, _ = replace_masked_values(v_bi, hypothesis_mask.unsqueeze(-1), -1e7).max(dim=1)

            v_a_avg = torch.sum(v_ai * premise_mask.unsqueeze(-1), dim=1) / torch.sum(premise_mask, 1, keepdim=True)
            v_b_avg = torch.sum(v_bi * hypothesis_mask.unsqueeze(-1), dim=1) / torch.sum(hypothesis_mask, 1, keepdim=True)

            # Now concat
            # (batch_size, (model_dim + k_dim) * 4)
            v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

            # the final MLP -- apply dropout to input, and MLP applies to output & hidden
            if self.dropout:
                v = self.dropout(v)

            output_hidden = self._output_feedforward(v)
            logit = self._output_logit(output_hidden)
            assert logit.size(-1) == 1
            label_logits.append(logit)

        label_logits = torch.cat(label_logits, -1)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ESIM':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        projection_feedforward = FeedForward.from_params(params.pop('projection_feedforward'))
        # inference_encoder = Seq2SeqEncoder.from_params(params.pop("inference_encoder"))
        output_feedforward = FeedForward.from_params(params.pop('output_feedforward'))
        output_logit = FeedForward.from_params(params.pop('output_logit'))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        dropout = params.pop("dropout", 0.3)
        hidden_size = params.pop("hidden_size", 300)
        add_lm_km = params.pop("add_lm_km", "modeling")
        lm_km_name = params.pop("lm_km_name", "bert-base-uncased")
        lm_km_type = params.pop("lm_km_type", "encoder")
        lm_km_act_mc = params.pop("lm_km_act_mc", "relu")
        lm_km_add_bias = params.pop("lm_km_add_bias", True)
        lm_km_q_map = params.pop("lm_km_q_map", "linear")

        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   encoder=encoder,
                   similarity_function=similarity_function,
                   projection_feedforward=projection_feedforward,
                   # inference_encoder=inference_encoder,
                   output_feedforward=output_feedforward,
                   output_logit=output_logit,
                   initializer=initializer,
                   dropout=dropout,
                   hidden_size=hidden_size,
                   add_lm_km=add_lm_km,
                   lm_km_name=lm_km_name,
                   lm_km_type=lm_km_type,
                   lm_km_act_mc=lm_km_act_mc,
                   lm_km_add_bias=lm_km_add_bias,
                   lm_km_q_map=lm_km_q_map,
                   regularizer=regularizer)

import argparse
from time import strftime, gmtime
import pickle
import io

import torch

from analysis.AttentionAnalyzer import AttentionAnalyzer
from model.data import SQuAD
from model.lmkm import PLMKM
from model.model import BiDAF


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)  # character embedding dimension
    parser.add_argument('--char-channel-width', default=5, type=int)  # character embedding CNN kernel size
    parser.add_argument('--char-channel-size', default=100, type=int)  # character embedding CNN channel number
    parser.add_argument('--context-threshold', default=400, type=int)  # maximum length of context or query
    parser.add_argument('--dev-batch-size', default=16, type=int)
    parser.add_argument('--dev-file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=15, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)  # EMA algorithm decay rate
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)  # frequency of printing validation results
    parser.add_argument('--train-batch-size', default=16, type=int)
    parser.add_argument('--train-file', default='train-v1.1.json')
    parser.add_argument('--word-dim', default=100, type=int)  # embedding and hidden size
    # ['modeling', 'none', 'random'], 'output' is not optimal
    parser.add_argument('--add-lm-km', default='modeling', type=str)
    # [bert-base-uncased, roberta-base, gpt2, facebook/bart-base]
    parser.add_argument('--lm-km-name', default='bert-base-uncased', type=str)
    parser.add_argument('--lm-km-type', default='encoder', type=str)  # [encoder, decoder, both]
    # feed-forward layers: relu activation + bias, or softmax without bias (not optimal)
    parser.add_argument('--lm-km-act-mc', default='relu', type=str)  # ['relu'], 'softmax' is not optimal
    parser.add_argument('--lm-km-add-bias', default=True, type=bool)  # add bias for relu
    # networks for transferring lstm hidden states to query vectors of knowledge memory
    parser.add_argument('--lm-km-q-map', default='linear', type=bool)  # ['linear'], 'mlp' is not optimal

    # Following params are for attention analysis.
    parser.add_argument('--model-path', type=str, default='saved_models/bert.pt', help='Model saved path.')
    parser.add_argument('--ema-path', type=str, default='saved_models/bert_ema.pkl', help='EMA saved path.')
    parser.add_argument('--attn-tensor-save-path', type=str, default='saved_models/bert_attention_score_wrt_layer.pt',
                        help='EMA saved path.')

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load ema
    with open(args.ema_path, 'rb') as f:
        ema = CPU_Unpickler(f).load()
    print('Finish loading pickle data...')

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    # setattr(args, 'dataset_file', f'data/squad/{args.dev_file}')
    setattr(args, 'dataset_file', f'data/squad/dev-v1.1.json')

    setattr(args, 'prediction_file', f'prediction{args.gpu}.out')
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    print('data loading complete!')

    print('loading pretrained language model...')
    lm_km = PLMKM(args.lm_km_name, args.lm_km_type)
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
    print('finish loading pretrained language model!')
    
    model = BiDAF(args, data.WORD.vocab.vectors, ffn_weights, active_layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    analyzer = AttentionAnalyzer(model=model, data=data)
    analyzer.attention_score(ema=ema, args=args)


if __name__ == '__main__':
    main()




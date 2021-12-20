"""
@Project  : MLProject2Attention
@File     : plot_heatmap.py
@Author   : Shaobo Cui
@Date     : 12/15/21 12:54 PM
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def normalized_attns(attns_file):
    attention_score = torch.load(attns_file, map_location=torch.device('cpu'))

    normalized_attention_score = (attention_score - torch.min(attention_score))
    normalized_attention_score = normalized_attention_score / torch.max(normalized_attention_score)
    return normalized_attention_score


if __name__ == '__main__':
    bert_attns_file = 'saved_models/bert_attention_score_wrt_layer.pt'
    random_attns_file = 'saved_models/random_attention_score_wrt_layer.pt'

    bert_normalized_attention_score = normalized_attns(bert_attns_file)
    random_normalized_attention_score = normalized_attns(random_attns_file)

    bert_normalized_attention_score = torch.flip(bert_normalized_attention_score, dims=[0])
    random_normalized_attention_score = torch.flip(random_normalized_attention_score, dims=[0])
    num_samples = random_normalized_attention_score.size(1)
    print('Shape: {}'.format(bert_normalized_attention_score.shape))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(22, 10, forward=True)
    fig.set_dpi(100)

    sns.set_theme()
    sns.set(font_scale=2.5)

    plt.tight_layout()
    y_axis_labels = range(12, 0, -1)
    sns.heatmap(random_normalized_attention_score, ax=ax1, cbar=True)
    sns.heatmap(bert_normalized_attention_score, ax=ax2, cbar=True)

    ax1.set_xticks([0, 2000, 17000, 32000, 34000])
    ax1.set_xticklabels(['1', '2', '...', 'N-1', 'N'], fontsize=25, rotation=0)
    ax1.set_yticklabels(y_axis_labels, fontsize=25)
    ax1.xaxis.set_tick_params(length=0)

    ax2.set_xticks([0, 2000, 17000, 32000, 34000])
    ax2.set_xticklabels(['1', '2', '...', 'N-1', 'N'], fontsize=25, rotation=0)
    ax2.set_yticklabels(y_axis_labels, fontsize=25)
    ax2.xaxis.set_tick_params(length=0)

    # plt.show()
    plt.savefig("heatmap.pdf")

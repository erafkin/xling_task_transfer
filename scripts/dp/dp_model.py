import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass, field
from typing import  Optional, Dict, List
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, TrainingArguments, Trainer
from transformers.trainer_utils import EvalLoopOutput
from tqdm import tqdm
import numpy as np

"""
    The goal is to have transformer base --> something that predicts arcs and relations which attend to each other. 
    Building off of the successful Biaffine DP model

"""

# Credit: taken from https://github.com/cambridgeltl/composable-sft/blob/main/examples/dependency-parsing/dp/modeling_biaffine.py
# Further credits in each model.

class Biaffine(nn.Module):
    # Class taken from https://github.com/yzhangcs/biaffine-parser
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.init_weights()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def init_weights(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        return s


class TransformerForBiaffineParsing(nn.Module):
    """
    Credit: G. Glavaš & I. Vulić
    Based on paper "Is Supervised Syntactic Parsing Beneficial for Language Understanding? An Empirical Investigation"
    (https://arxiv.org/pdf/2008.06788.pdf)
    """

    def __init__(self,  encoder: nn.Module, num_labels: int, dropout: float = 0.1, bert:bool=False):
        super().__init__()
        self.roberta = None
        self.bert = None
        self.encoder_hidden_size = None
        if bert:
            self.bert = encoder
            self.encoder_hidden_size = self.bert.config.hidden_size
            self.pad_token_id = self.bert.config.pad_token_id
        else:
            self.roberta = encoder
            self.classifier = self.roberta.config.hidden_size
            self.pad_token_id = self.roberta.config.pad_token_id

        self.biaffine_arcs = Biaffine(n_in=self.encoder_hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(n_in=self.encoder_hidden_size, n_out=num_labels, bias_x=True, bias_y=True)
        self.dropout = nn.Dropout(dropout)
        self.loss_fn = CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels_arcs=None,
        labels_rels=None,
        word_starts=None,
    ):

        # run through encoder and get vector representations
        if self.bert is None:
            out_trans = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        else:
            out_trans = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
        )
        

        outs = self.dropout(out_trans[0])

        word_outputs_deps = self._merge_subword_tokens(outs, word_starts)
        # adding the CLS representation as the representation for the "root" parse token -- will this be a problem for roberta?
        word_outputs_heads = torch.cat([out_trans[0], word_outputs_deps], dim=1)

        arc_preds = self.biaffine_arcs(word_outputs_deps, word_outputs_heads)
        arc_preds = arc_preds.squeeze()
        outputs = (arc_preds,)

        rel_preds = self.biaffine_rels(word_outputs_deps, word_outputs_heads)
        rel_preds = rel_preds.permute(0, 2, 3, 1)
        outputs = (rel_preds,) + outputs

        loss = self._get_loss(arc_preds, rel_preds, labels_arcs, labels_rels, self.loss_fn)

        outputs = (loss,) + outputs

        if len(arc_preds.shape) == 2:
            return loss, rel_preds, arc_preds.unsqueeze(0)
        return outputs # should be loss, rel_preds, arc_preds

    def _merge_subword_tokens(self, subword_outputs, word_starts):
        instances = []
        max_seq_length = subword_outputs.shape[1]

        # handling instance by instance
        for i in range(len(subword_outputs)):
            subword_vecs = subword_outputs[i]
            word_vecs = []
            starts = word_starts[i]
            mask = starts.ne(self.pad_token_id)
            starts = starts[mask]
            for j in range(len(starts) - 1):
                if starts[j + 1] <= 0:
                    break

                start = starts[j]
                end = starts[j + 1]
                vecs_range = subword_vecs[start:end]
                word_vecs.append(torch.mean(vecs_range, 0).unsqueeze(0))

            instances.append(word_vecs)

        t_insts = []
        zero_tens = torch.zeros(self.encoder_hidden_size).unsqueeze(0)
        zero_tens = zero_tens.to("cuda" if torch.cuda.is_available() else "cpu")

        for inst in instances:
            if len(inst) < max_seq_length:
                for i in range(max_seq_length - len(inst)):
                    inst.append(zero_tens)
            t_insts.append(torch.cat(inst, dim=0).unsqueeze(0))

        w_tens = torch.cat(t_insts, dim=0)
        return w_tens

    def _get_loss(self, arc_preds, rel_preds, labels_arc, labels_rel, loss_fn):
        if len(arc_preds.shape) == 2:
            arc_preds = arc_preds.unsqueeze(0)

        mask = labels_arc.ne(self.pad_token_id)
        arc_scores, arcs = arc_preds[mask], labels_arc[mask]
        loss = loss_fn(arc_scores, arcs)

        rel_scores, rels = rel_preds[mask], labels_rel[mask]
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]
        rel_loss = loss_fn(rel_scores, rels)
        loss += rel_loss

        return loss
    


@dataclass
class DataCollatorForDependencyParsing:
    # Taken from https://github.com/cambridgeltl/composable-sft/blob/main/examples/dependency-parsing/dp/utils_udp.py
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None

    def __call__(self, features):
        for f in features:
            for k in f:
                if isinstance(f[k], torch.Tensor):
                    f[k] = f[k].tolist()
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
        )

        seq_len = len(batch['input_ids'][0])
        for k in ['labels_arcs', 'labels_rels', 'word_starts']:
            if k in batch:
                for i, example in enumerate(batch[k]):
                    if self.tokenizer.padding_side == 'right':
                        if isinstance(example, torch.Tensor):
                            example = example.tolist()
                        example +=  (seq_len - len(example)) * [self.tokenizer.pad_token_id]
                    else:
                        batch[k][i] = example + (seq_len - len(example)) * [self.tokenizer.pad_token_id]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch
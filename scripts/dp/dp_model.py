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
                        example += (seq_len - len(example)) * [self.tokenizer.pad_token_id]
                    else:
                        batch[k][i] = example + (seq_len - len(example)) * [self.tokenizer.pad_token_id]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch



class Metric(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def unpack(*tensors: torch.Tensor):
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


class ParsingMetric(Metric):
    """
    based on allennlp.training.metrics.AttachmentScores
    Computes labeled and unlabeled attachment scores for a dependency parse. Note that the input
    to this metric is the sampled predictions, not the distribution itself.
    """

    def __init__(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0

    def add(
        self,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
    ):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        """
        unwrapped = self.unpack(predicted_indices, predicted_labels, gold_indices, gold_labels)
        predicted_indices, predicted_labels, gold_indices, gold_labels = unwrapped

        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        correct_indices = predicted_indices.eq(gold_indices).long()
        correct_labels = predicted_labels.eq(gold_labels).long()
        correct_labels_and_indices = correct_indices * correct_labels

        self._unlabeled_correct += correct_indices.sum().item()
        self._labeled_correct += correct_labels_and_indices.sum().item()
        self._total_words += correct_indices.numel()

    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0


@dataclass
class UDTrainingArguments(TrainingArguments):
    """
    Extends TrainingArguments for Universal Dependencies (UD) dependency parsing.
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    decode_mode: str = field(default="greedy", metadata={"help": "Whether to use mst decoding or greedy decoding"})
    metric_score: Optional[str] = field(
        default=None, metadata={"help": "Metric used to determine best model during training."}
    )

class DependencyParsingTrainer(Trainer):
    args: UDTrainingArguments

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys=None,
        metric_key_prefix=None,
    ) -> EvalLoopOutput:
        """
                Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

                Works both with or without labels.
                """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        eval_losses: List[float] = []
        model.eval()

        metric = ParsingMetric()

        for inputs in tqdm(dataloader, desc=description):

            step_eval_loss, (rel_preds, arc_preds), _ = self.prediction_step(model, inputs, False)

            eval_losses += [step_eval_loss.mean().item()]

            mask = inputs["labels_arcs"].ne(self.model.config.pad_token_id)
            predictions_arcs = torch.argmax(arc_preds, dim=-1)[mask]

            labels_arcs = inputs["labels_arcs"][mask]

            predictions_rels, labels_rels = rel_preds[mask], inputs["labels_rels"][mask]
            predictions_rels = predictions_rels[torch.arange(len(labels_arcs)), labels_arcs]
            predictions_rels = torch.argmax(predictions_rels, dim=-1)

            metric.add(labels_arcs, labels_rels, predictions_arcs, predictions_rels)

        results = metric.get_metric()
        results = {
            k if k.startswith(metric_key_prefix) else f'{metric_key_prefix}_{k}': v
            for k, v in results.items()
        }
        results[f"{metric_key_prefix}_loss"] = np.mean(eval_losses)

        # Add predictions_rels to output, even though we are only interested in the metrics
        return EvalLoopOutput(
            predictions=predictions_rels,
            label_ids=None,
            metrics=results,
            num_samples=self.num_examples(dataloader),
        )
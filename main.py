from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

from absl import logging
from absl import app

import tensorflow as tf
from transformers import BertTokenizer

import config
from preprocess import ProcessFactory
from joint_model import JointCategoricalBert
from data_loader import load_tf_features

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}

parser = argparse.ArgumentParser()

parser.add_argument("--task", default="atis", type=str, help="The name of the task to train")
parser.add_argument("--model_dir", default="atis_model", type=str, help="Path to save, load model")
parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

parser.add_argument("--model_type", default="bert", type=str,
                    help="Model type selected in the list: ")

parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
parser.add_argument("--max_seq_len", default=50, type=int,
                    help="The maximum total input sequence length after tokenization.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs", default=10.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", default=True, action="store_true", help="Whether to run eval on the test set.")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

parser.add_argument("--ignore_index", default=0, type=int,
                    help='Specifies a target value that is ignored and does not contribute to the input gradient')

parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

# CRF option
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
parser.add_argument("--slot_pad_label", default="PAD", type=str,
                    help="Pad token for slot label pad (to be ignore when calculate loss)")

args = parser.parse_args()

args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

def read_file(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            lines.append(line.strip())
        return lines

def main(argv):
    """Main function for training process.
    """
    del argv

    tf.config.experimental_run_functions_eagerly(config.tf_eager_execution)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    #
    # data_factory = ProcessFactory(
    #     sentences=config.sentences_file,
    #     intents=config.intents_file,
    #     slots=config.slots_file,
    #     split=config.validation_set_ratio)
    # data = data_factory.get_data()
    logging.info('after preprocess')

    dev_features = load_tf_features(args, tokenizer, mode="dev")
    train_features = load_tf_features(args, tokenizer, mode="train")

    intents = read_file("data/atis/intent_label.txt")
    slots = read_file("data/atis/slot_label.txt")

    model = JointCategoricalBert(
        train=[],
        validation=[],
        intents_num=len(intents),
        slots_num=len(slots))

    model.fit_features(train_features, dev_features)

    # model = JointCategoricalBert(
    #     train=data['train'],
    #     validation=data['validation'],
    #     intents_num=data_factory.get_intents_num(),
    #     slots_num=data_factory.get_slots_num())
    logging.info('after initializing model')

    model.fit()
    model.get_model().summary()
    model.save_model("nlu")

    # slot_logits, intent_logits = model.get_model().predict(data['validation'].get_tokens())


if __name__ == '__main__':
    app.run(main)

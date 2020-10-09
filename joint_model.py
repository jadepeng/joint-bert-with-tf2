from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from absl import logging
import tensorflow as tf
from tensorflow.python.keras.layers import TimeDistributed, Dense, Bidirectional, LSTM
from transformers import TFBertModel, BertTokenizer, AdamWeightDecay

import config
from preprocess import Process


class CustomBertLayer(tf.keras.layers.Layer):
    """Custom layer to modify build and call methods of BERT if needed.
    """

    def __init__(self, **kwargs):
        super(CustomBertLayer, self).__init__(**kwargs)
        self._bert = self._load_bert()

    def _load_bert(self):
        model = TFBertModel.from_pretrained(config.bert_model_name)

        logging.info('BERT weights loaded')
        return model

    def build(self, input_shape):
        super(CustomBertLayer, self).build(input_shape)

    def call(self, inputs):
        result = self._bert(inputs=inputs)

        return result


class CustomModel(tf.keras.Model):
    """Definition of the model to modify with custom call method.

    Args:
        intents_num(int):
            Number of intents in the working dataset that used in softmax layer.
        slots_num(int):
            Number of slots labels in the working dataset that used in softmax layer.
    """

    def __init__(self,
                 intents_num: int,
                 slots_num: int):
        super().__init__(name="joint_intent_slot")
        self._bert_layer = CustomBertLayer()
        self._dropout = tf.keras.layers.Dropout(rate=config.dropout_rate)
        self._intent_classifier = tf.keras.layers.Dense(intents_num,
                                                        activation='softmax',
                                                        name='intent_classifier')
        # self._intent_gru = Bidirectional(LSTM(128, dropout=0.1), name="bi-lstm")
        self._slot_classifier = TimeDistributed(Dense(slots_num,
                                                      activation='softmax',
                                                      name='slot_classifier'), name="slot_time_distributed")
        self._intent_dense = tf.keras.layers.Dense(units=768, activation="tanh")
        self.intents_num = intents_num
        self.slots_num = slots_num

    def call(self, inputs, training=False, **kwargs):
        # bert 编码 （序列向量，池化向量（句向量））
        # (?, 768)
        # (?, max_length, 768)

        # x: {input_ids，attention_mask}
        # y: {slots, intents}
        #
        # slots:  [[0,0,1,2,...],[],[]]  multi_hot
        # intents: [0,1,2,3,2,3] onehot

        sequence_output, pooled_output = self._bert_layer(inputs, **kwargs)

        sequence_output = self._dropout(sequence_output, training)
        slot_logits = self._slot_classifier(sequence_output)

        pooled_output = self._dropout(pooled_output, training)
        intent_logits = self._intent_dense(pooled_output)
        intent_logits = self._dropout(intent_logits)
        # intent_logits = tf.keras.layers.Dense(units=len(classes), activation="softmax")(intent_logits)

        # pooled_output = self._intent_gru(pooled_output, training)
        intent_logits = self._intent_classifier(intent_logits)

        return slot_logits, intent_logits

class IntentModel(tf.keras.Model):
    """Definition of the model to modify with custom call method.

    Args:
        intents_num(int):
            Number of intents in the working dataset that used in softmax layer.
        slots_num(int):
            Number of slots labels in the working dataset that used in softmax layer.
    """

    def __init__(self,
                 intents_num: int):
        super().__init__(name="joint_intent_slot")
        self._bert_layer = CustomBertLayer()
        self._dropout = tf.keras.layers.Dropout(rate=config.dropout_rate)
        self._intent_classifier = tf.keras.layers.Dense(intents_num,
                                                        activation='softmax',
                                                        name='intent_classifier')
        self._intent_dense = tf.keras.layers.Dense(units=768, activation="tanh")
        self.intents_num = intents_num

    def call(self, inputs, training=False, **kwargs):
        # bert 编码 （序列向量，池化向量（句向量））
        # (?, 768)
        # (?, max_length, 768)

        # x: {input_ids，attention_mask}
        # y: {slots, intents}
        #
        # slots:  [[0,0,1,2,...],[],[]]  multi_hot
        # intents: [0,1,2,3,2,3] onehot

        sequence_output, pooled_output = self._bert_layer(inputs, **kwargs)


        pooled_output = self._dropout(pooled_output, training)
        intent_logits = self._intent_dense(pooled_output)
        intent_logits = self._dropout(intent_logits)
        # intent_logits = tf.keras.layers.Dense(units=len(classes), activation="softmax")(intent_logits)

        # pooled_output = self._intent_gru(pooled_output, training)
        intent_logits = self._intent_classifier(intent_logits)

        return intent_logits


class IntentCategoricalBert(object):
    """Wrapper to model functions. The Model compiles with hyper-parameters and
    will be ready for fit.

    Args:
        train(preprocess.Process):
            Holds the training part of samples.
        validation(preprocess.Process):
            Holds the validation part of samples.
        intents_num(int):
            Number of intents in the working dataset which will be used in softmax layer.
        slots_num(int):
            Number of slot lables in the working dataset which will be used in softmax layer.
    """

    def __init__(self,
                 intents_num: int):
        self._model = IntentModel(intents_num=intents_num)
        self._compile()

    def _compile(self):
        """Compile the model with hyper-parameters that defined in the config file.
        """
        #  如果你的 targets 是 one-hot 编码，用 categorical_crossentropy
        # 　　one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
        # 如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy
        # 　　数字编码：2, 0, 1

        optimizer = AdamWeightDecay(learning_rate=config.learning_rate, epsilon=1e-8)
        losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        self._model.compile(optimizer=optimizer,
                            loss=losses,
                            metrics=metrics)
        logging.info("model compiled")

    def get_model(self):
        return self._model

    def save_model(self, model_file):
        self._model.save_weights(model_file)

    def load_model(self, model_file):
        self._model.load_weights(model_file)


class JointCategoricalBert(object):
    """Wrapper to model functions. The Model compiles with hyper-parameters and
    will be ready for fit.

    Args:
        train(preprocess.Process):
            Holds the training part of samples.
        validation(preprocess.Process):
            Holds the validation part of samples.
        intents_num(int):
            Number of intents in the working dataset which will be used in softmax layer.
        slots_num(int):
            Number of slot lables in the working dataset which will be used in softmax layer.
    """

    def __init__(self,
                 train: Process,
                 validation: Process,
                 intents_num: int,
                 slots_num: int):
        self._dataset = {'train': train, 'validation': validation}
        self._model = CustomModel(intents_num=intents_num, slots_num=slots_num)
        self._compile()

    def _compile(self):
        """Compile the model with hyper-parameters that defined in the config file.
        """
        #  如果你的 targets 是 one-hot 编码，用 categorical_crossentropy
        # 　　one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
        # 如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy
        # 　　数字编码：2, 0, 1

        optimizer = AdamWeightDecay(learning_rate=config.learning_rate, epsilon=1e-8)
        losses = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)]
        loss_weights = [config.loss_weights['slot'], config.loss_weights['intent']]
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        self._model.compile(optimizer=optimizer,
                            loss=losses,
                            loss_weights=loss_weights,
                            metrics=metrics)
        logging.info("model compiled")

    def fit_features(self, train_features,validation_features):
        """Fit the compiled model to the dataset. Hyper-parameters such as number of
        epochs defined in the config file.
        """
        logging.info('before fit model')

        self._model.fit(
            {
                "input_ids": train_features[0],
                "attention_mask": train_features[1],
                "token_type_ids": train_features[2]
            },
            (train_features[3], train_features[4]),
            validation_data=(
                {
                    "input_ids": validation_features[0],
                    "attention_mask": validation_features[1],
                    "token_type_ids": validation_features[2]
                },
                (validation_features[3], validation_features[4])),
            epochs=config.epochs_num,
            batch_size=config.batch_size)

        return self._model

    def fit(self):
        """Fit the compiled model to the dataset. Hyper-parameters such as number of
        epochs defined in the config file.
        """
        logging.info('before fit model')

        self._model.fit(
            self._dataset['train'].get_tokens(),
            (self._dataset['train'].get_slots(), self._dataset['train'].get_intents()),
            validation_data=(
                self._dataset['validation'].get_tokens(),
                (self._dataset['validation'].get_slots(),
                 self._dataset['validation'].get_intents())),
            epochs=config.epochs_num,
            batch_size=config.batch_size)

        return self._model

    def get_model(self):
        return self._model

    def save_model(self, model_file):
        self._model.save_weights(model_file)

    def load_model(self, model_file):
        self._model.load_weights(model_file)


if __name__ == '__main__':
    sentence = 'i would like a flight traveling one way from phoenix to san diego on april first'
    model = TFBertModel.from_pretrained(config.bert_model_name)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    sequence_dict = tokenizer.encode_plus(sentence,
                                          add_special_tokens=True,
                                          max_length=50,
                                          pad_to_max_length=True,
                                          truncation=True)
    sequence_output, pooled_output = model([np.array([sequence_dict['input_ids']]),
                                            np.array([sequence_dict['token_type_ids']]),
                                            np.array([sequence_dict['attention_mask']])])
    print(sequence_output)
    print(pooled_output)
    print(sequence_output.shape)
    print(pooled_output.shape)

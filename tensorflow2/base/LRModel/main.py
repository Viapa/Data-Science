import os
import sys
import json
import argparse
import datetime

import tensorflow as tf
from dmlite.tensorflow import save_model, TFRecordReader

from models import MyModel



def main():
    parser = argparse.ArgumentParser(description='Package purchase cvr seq-model training')
    parser.add_argument('--trainInputs', type=str, help='input train data path, this name required')
    parser.add_argument('--valInputs', type=str, help='input validation data path, this name required')
    parser.add_argument('--saveDir', type=str, help='output logs and models directory, this name required')
    parser.add_argument('--featureCols', type=str, required=False, default='', help='comma-separated feature column names, this name required')
    parser.add_argument('--labelCols', type=str, required=False, default='', help='comma-separated label column names, this name required')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
    parser.add_argument("--per_worker_batch_size", type=int, default=2048, metavar="N", required=False)
    args = parser.parse_args()
    print("-----------------Input Args-----------------------")
    print(args)
    print("-----------------GPU Devices---------------------")
    print(tf.config.experimental.list_physical_devices('GPU'))
    # 数据载入
    feature_cols = [col_name for col_name in args.featureCols.split(',')]
    label_cols = [col_name for col_name in args.labelCols.split(',')]
    num_features = len(feature_cols)
    num_labels = len(label_cols)
    print("Number of features:", num_features)
    print("Number of labels:", num_labels)
    # 数据处理
    train_reader = TFRecordReader(input_path=args.trainInputs,
                                  feature_cols=feature_cols,
                                  label_cols=label_cols,
                                  batch_size=args.per_worker_batch_size,
                                  shuffle=True,
                                  shuffle_buffer_size=200000,
                                  buffer_size=100000000)
    valid_reader = TFRecordReader(input_path=args.valInputs,
                                  feature_cols=feature_cols,
                                  label_cols=label_cols,
                                  batch_size=args.per_worker_batch_size,
                                  shuffle=False,
                                  buffer_size=100000000)
    train_ds = train_reader.dataset().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    valid_ds = valid_reader.dataset().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    print("Prepare Dataset Success!")

    # 训练参数
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    metrics = [tf.metrics.AUC()]
    # 模型准备
    feature_num = 271
    hidden_size = 768
    model = MyModel(hidden_size)
    model.build(input_shape=(None, feature_num))
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    print(model.summary())
    # 模型训练
    print("=====" * 8 + "%s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Training Started!")
    history = model.fit(train_ds,
                        validation_data=valid_ds,
                        epochs=args.epochs,
                        verbose=2)
    print("=====" * 8 + "%s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("Training Success!")
    # 模型评估
    loss, auc = model.evaluate(valid_ds)
    print("Loss & AUC on validation dataset: loss = %.4f, auc = %.4f" % (loss, auc))
    # 模型保存
    print('Start save model...')
    save_model(model,
               output_dir=args.saveDir,
               input_names=['input_1'],
               input_types=['DT_FLOAT'],
               inputShapes=[[-1, feature_num]]
               )
    print("Save Model Success!")


if __name__ == '__main__':
    main()

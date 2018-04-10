import numpy as np
import os
import tensorflow as tf
import argparse
import math

from tensorflow.examples.tutorials.mnist import input_data
from logger import Logger
from fast_gradient import fgm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas_equal",  action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=200)    
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    mnist_data = input_data.read_data_sets('/tmp/mnistdata', validation_size=0)
    fmt = {"IZY": '.2f', "IXZ_1": '.2f', "IZ_1Z_2": '.2f', "IZ_2Z_3": '.2f', 
           "acc": '.4f', "avg_acc": '.4f', "err": '.4f', "avg_err": '.4f', "adv_acc": '.4f', "avg_adv_acc": '.4f'}
    
    if args.debug:
        betas = [1e-2]
        num_epochs = 1
    else:
        betas = np.logspace(-9, 0, 10)
        num_epochs = args.num_epochs
    settings = []
    for beta in betas:
        for use_stoch in [[True, True, True], [False, False, True], [False, True, True]]:
            setting = {
                "num_epochs": num_epochs,
                "use_stoch": use_stoch,
                "betas": [beta] * 3,
            }
            settings.append(setting)
            
    for setting in settings:
        str_setting = "_".join(map(str, [setting["num_epochs"]] + setting["use_stoch"] + setting["betas"]))
        logger = Logger("multi_zetas" + str(str_setting), fmt=fmt)
        model_base_dir = os.path.join("./models", str_setting)

        def make_experiment(num_epochs=200, stoch_z_dims=[512, 256, 128],
                            use_stoch=[True, True, True], betas=None, betas_equal=True):
            tf.reset_default_graph()
            assert(len(stoch_z_dims) == len(use_stoch))
            default_betas = np.array([1. / x for x in stoch_z_dims])

            if not betas_equal:
                betas *= default_betas

            if not betas:
                betas = default_betas

            num_stoch_z = len(stoch_z_dims)
            images = tf.placeholder(tf.float32, [None, 784], 'images')
            labels = tf.placeholder(tf.int64, [None], 'labels')
            one_hot_labels = tf.one_hot(labels, 10)

            layers = tf.contrib.layers
            ds = tf.contrib.distributions

            def encoder(x, stoch_dim, use_stoch=True, first=False):
                if first:
                    x = 2 * x - 1
                net = layers.relu(x, 2 * stoch_dim)
                net = layers.relu(net, 2 * stoch_dim)
                params = layers.linear(net, 2 * stoch_dim)
                if use_stoch:
                    mu, rho = params[:, :stoch_dim], params[:, stoch_dim:]
                    encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
                    return encoding
                return tf.nn.relu(params)

            def decoder(encoding_sample):
                net = layers.linear(encoding_sample, 10)
                return net

            prior = ds.Normal(0.0, 1.0)
            x = images
            encoding = None
            info_losses = []
            for i in range(num_stoch_z):
                with tf.variable_scope('encoder_' + str(i)):
                    if encoding is not None:
                        if use_stoch[i - 1]:
                            x = encoding.sample()
                        else:
                            x = encoding
                    first = True if i == 0 else False
                    encoding = encoder(x, stoch_z_dims[i], use_stoch[i], first)
                    if use_stoch[i]:
                        info_losses.append(ds.kl_divergence(encoding, prior))

            with tf.variable_scope('decoder'):
                last_z = encoding.sample() if use_stoch[-1] else encoding
                logits = decoder(last_z)

            with tf.variable_scope('decoder', reuse=True):
                last_z = encoding.sample(12) if use_stoch[-1] else encoding
                many_logits = decoder(last_z)

            def model(x, num_examples=1, logits=False):
                encoding = None
                for i in range(num_stoch_z):
                    with tf.variable_scope('encoder_' + str(i), reuse=True):
                        if encoding is not None:
                            if use_stoch[i - 1]:
                                x = encoding.sample()
                            else:
                                x = encoding
                        first = True if i == 0 else False
                        encoding = encoder(x, stoch_z_dims[i], use_stoch[i], first)

                last_z = encoding.sample() if use_stoch[-1] else encoding
                with tf.variable_scope('decoder', reuse=True):
                    logits_ = decoder(last_z)
                y = tf.nn.softmax(logits_)
                if logits:
                    return y, logits_
                return y

            adv_examples = fgm(model, images, eps=0.35)

            class_loss = tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=one_hot_labels) / math.log(2)

            IZY_bound = math.log(10, 2) - class_loss
            IZX_bounds = []

            cum_info_loss = 0.0
            for i, info_loss in enumerate(info_losses):
                cur_bound = tf.reduce_sum(tf.reduce_mean(info_loss, 0))
                IZX_bounds.append(cur_bound)
                cum_info_loss += cur_bound * betas[i]

            total_loss = class_loss + cum_info_loss

            accuracy = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(logits, 1), labels), tf.float32))
            avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))

            summary_path = os.path.join(model_base_dir, "./summaries")
            train_writer = tf.summary.FileWriter(os.path.join(summary_path, './train'), flush_secs=60,
                                                 graph=tf.get_default_graph())
            test_writer = tf.summary.FileWriter(os.path.join(summary_path, './test'), flush_secs=60,
                                                 graph=tf.get_default_graph())

            tf.summary.scalar("IZX", IZX_bounds[0])
            tf.summary.scalar("IZY", IZY_bound)
            tf.summary.scalar("accuracy", accuracy)
            tf.summary.scalar("average_accuracy", avg_accuracy)
            merged = tf.summary.merge_all()

            batch_size = 100
            steps_per_batch = int(mnist_data.train.num_examples / batch_size)

            global_step = tf.contrib.framework.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                                       decay_steps=2*steps_per_batch,
                                                       decay_rate=0.97, staircase=True)
            opt = tf.train.AdamOptimizer(learning_rate, 0.5)

            ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
            ma_update = ma.apply(tf.model_variables())

            saver = tf.train.Saver()
            saver_polyak = tf.train.Saver(ma.variables_to_restore())

            train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                               global_step,
                                                               update_ops=[ma_update])
            with tf.Session() as sess:
                tf.global_variables_initializer().run()

                def evaluate(num_epoch, flag="test"):
                    feed_dict = {images: mnist_data.test.images, labels: mnist_data.test.labels}
                    if flag == "train":
                        feed_dict = {images: mnist_data.train.images, labels: mnist_data.train.labels}
                    adv_acc, avg_adv_acc = 0, 0
                    avg_metric = avg_accuracy if use_stoch[-1] else accuracy
                    if flag == "test":
                        fgm_examples = sess.run(adv_examples, feed_dict=feed_dict)
                        adv_acc, avg_adv_acc = sess.run([accuracy, avg_metric], feed_dict={images: fgm_examples, 
                                                                                labels: mnist_data.test.labels})
                    summary, IZY, IZX_s, acc, avg_acc = sess.run([merged, IZY_bound, IZX_bounds, accuracy, avg_metric],
                                             feed_dict=feed_dict)
                    if flag == "test":
                        test_writer.add_summary(summary, num_epoch)

                    return IZY, IZX_s, acc, avg_acc, 1 - acc, 1 - avg_acc, adv_acc, avg_adv_acc

                for epoch in range(1, num_epochs + 1):
                    for step in range(steps_per_batch):
                        im, ls = mnist_data.train.next_batch(batch_size)
                        summary, _ = sess.run([merged, train_tensor], feed_dict={images: im, labels: ls})
                        if step == steps_per_batch - 1:
                            train_writer.add_summary(summary, epoch)

                    metrics = evaluate(epoch)
                    logger.add_scalar(epoch, 'IZY', metrics[0])
                    for i in range(len(metrics[1])):
                        if i == 0:
                            logger.add_scalar(epoch, "IXZ_1", metrics[1][i])
                        else:
                            logger.add_scalar(epoch, "IZ_{}Z_{}".format(i, i + 1), metrics[1][i])

                    acc, avg_acc, err, avg_err, adv_acc, avg_adv_acc = metrics[2:]
                    logger.add_scalar(epoch, "acc", acc)
                    logger.add_scalar(epoch, "avg_acc", avg_acc)
                    logger.add_scalar(epoch, "err", err)
                    logger.add_scalar(epoch, "avg_err", avg_err)
                    logger.add_scalar(epoch, "adv_acc", adv_acc)
                    logger.add_scalar(epoch, "avg_adv_acc", avg_adv_acc)
                    logger.iter_info()

                    if args.early_stopping:
                        first, second = False, False
                        best_adv_acc = max(logger.scalar_metrics["adv_acc"], key=lambda x: x[1])
                        best_avg_adv_acc = max(logger.scalar_metrics["avg_adv_acc"], key=lambda x: x[1])
                        delta = 0.01
                        delta_1 = best_adv_acc[1] - logger.scalar_metrics["adv_acc"][-1][1]
                        if delta_1 > delta and best_adv_acc[0] - epoch > 10:
                            first = True
                        delta_2 = best_avg_adv_acc[1] - logger.scalar_metrics["avg_adv_acc"][-1][1]
                        if delta_2 > delta and best_avg_adv_acc[0] - epoch > 10:
                            second = True
                        if first and second:
                            break

                ckpts_path = os.path.join(model_base_dir, "ckpts")
                savepth = saver.save(sess, ckpts_path, global_step)        
                saver_polyak.restore(sess, savepth)
                logger.save()

        make_experiment(setting["num_epochs"], use_stoch=setting["use_stoch"], betas=setting["betas"])     


if __name__ == "__main__":
    main()

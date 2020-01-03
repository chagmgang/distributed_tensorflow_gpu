import tensorflow as tf
import numpy as np

import tensorboardX
import contextlib
import time

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'], 'Job name. Ignored when task is set to -1')

@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""
    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get('collections', None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope('', custom_getter=getter) as vs:
        yield vs

def main(_):
    num_actors = 1
    task = 0
    local_job_device = '/job:{}/task:{}'.format(FLAGS.job_name, task)
    shared_job_device = '/job:learner/task:0'
    is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == task
    is_learner = FLAGS.job_name == 'learner'

    global_variable_device = shared_job_device + '/cpu'

    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:{}'.format(8001+i) for i in range(num_actors)],
        'learner': ['localhost:8000']})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task)

    with tf.Graph().as_default(), \
         tf.device(local_job_device), \
         pin_global_variables(global_variable_device):

        ## build agent
        with tf.device(shared_job_device):
            with tf.device('/cpu'):
                ph = tf.placeholder(tf.float32, shape=[None, 4])
                l1 = tf.layers.dense(inputs=ph, units=256, activation=tf.nn.relu)
                for i in range(100):
                    l1 = tf.layers.dense(inputs=l1, units=256, activation=tf.nn.relu)
                l2 = tf.layers.dense(inputs=l1, units=3, activation=tf.nn.softmax)

        ## build learner
        if is_learner:
            with tf.device('/gpu'):
                ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
                label = tf.placeholder(tf.int32, shape=[None])
                onehot = tf.one_hot(label, 10)

                conv1 = tf.layers.conv2d(
                    inputs=ph, filters=32, kernel_size=[5, 5], padding='valid',
                    strides=[2, 2], activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(
                    inputs=conv1, filters=32, kernel_size=[5, 5], padding='valid',
                    strides=[2, 2], activation=tf.nn.relu)
                flatten = tf.layers.flatten(conv2)
                predict = tf.layers.dense(inputs=flatten, units=10, activation=tf.nn.softmax)

                cross_entropy = tf.reduce_mean(-onehot * tf.log(predict))
                train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

        else:
            with tf.device('/cpu'):
                ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
                label = tf.placeholder(tf.int32, shape=[None])
                onehot = tf.one_hot(label, 10)

                conv1 = tf.layers.conv2d(
                    inputs=ph, filters=32, kernel_size=[5, 5], padding='valid',
                    strides=[2, 2], activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(
                    inputs=conv1, filters=32, kernel_size=[5, 5], padding='valid',
                    strides=[2, 2], activation=tf.nn.relu)
                flatten = tf.layers.flatten(conv2)
                predict = tf.layers.dense(inputs=flatten, units=10, activation=tf.nn.softmax)

                cross_entropy = tf.reduce_mean(-onehot * tf.log(predict))
                train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

        sess = tf.Session(server.target)
        sess.run(tf.global_variables_initializer())

    ((train_data, train_labels),
     (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data/np.float32(255)
    train_data = np.expand_dims(train_data, axis=3)
    train_labels = train_labels.astype(np.int32)

    batch_size = 50
    data_size = len(train_data)
    epoch_time = int(data_size / batch_size)

    if is_learner:

        writer = tensorboardX.SummaryWriter('runs/gpu')
        for i in range(epoch_time):
            batch_data = train_data[i*batch_size : (i+1)*batch_size]
            batch_label = train_labels[i*batch_size : (i+1)*batch_size]

            start_time = time.time()
            loss, _ = sess.run(
                [cross_entropy, train],
                feed_dict={ph: batch_data, label: batch_label})
            end_time = time.time()

            print(end_time - start_time, loss)

            writer.add_scalar('data/time', end_time - start_time, i)
            writer.add_scalar('data/loss', loss, i)

    else:
        writer = tensorboardX.SummaryWriter('runs/cpu')
        for i in range(epoch_time):
            batch_data = train_data[i*batch_size : (i+1)*batch_size]
            batch_label = train_labels[i*batch_size : (i+1)*batch_size]

            start_time = time.time()
            loss, _ = sess.run(
                [cross_entropy, train],
                feed_dict={ph: batch_data, label: batch_label})
            end_time = time.time()

            print(end_time - start_time, loss)

            writer.add_scalar('data/time', end_time - start_time, i)
            writer.add_scalar('data/loss', loss, i)

if __name__ == '__main__':
    tf.app.run()
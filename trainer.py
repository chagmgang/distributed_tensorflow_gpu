import tensorflow as tf
import numpy as np

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

def model():
    ph = tf.placeholder(tf.float32, shape=[None, 4])
    result = tf.layers.dense(inputs=ph, units=3, activation=tf.nn.softmax)
    return ph, result

def build_learner(result):
    return tf.layers.dense(inputs=result, units=2, activation=tf.nn.softmax)

def main(_):
    num_actors = 1
    task = 0
    local_job_device = '/job:{}/task:{}'.format(FLAGS.job_name, task)
    shared_job_device = '/job:learner/task:0'
    is_actor_fn = lambda i: FLAGS.job_name == 'actor' and i == task
    is_learner = FLAGS.job_name == 'learner'
    filters = [shared_job_device, local_job_device]

    global_variable_device = shared_job_device + '/cpu'

    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:{}'.format(8001+i) for i in range(num_actors)],
        'learner': ['localhost:8000']})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task)
    
    with tf.device(local_job_device):

        if is_learner:
            with tf.device(shared_job_device):
                shared_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
                shared_conv1 = tf.layers.conv2d(
                    inputs=shared_input, filters=32, kernel_size=[5, 5], strides=[2, 2],
                    padding='valid', activation=tf.nn.relu)
                shared_conv2 = tf.layers.conv2d(
                    inputs=shared_conv1, filters=32, kernel_size=[5, 5], strides=[2, 2],
                    padding='valid', activation=tf.nn.relu)
                shared_flatten = tf.layers.flatten(shared_conv2)
                shared_flatten = tf.layers.dense(inputs=shared_flatten, units=256, activation=tf.nn.relu)
                shared_flatten = tf.layers.dense(inputs=shared_flatten, units=256, activation=tf.nn.relu)
                shared_flatten = tf.layers.dense(inputs=shared_flatten, units=256, activation=tf.nn.relu)
                shared_flatten = tf.layers.dense(inputs=shared_flatten, units=256, activation=tf.nn.relu)
                shared_predict = tf.layers.dense(inputs=shared_flatten, units=10, activation=tf.nn.softmax)

                shared_label = tf.placeholder(tf.int32, shape=[None])
                shared_onehot = tf.one_hot(shared_label, 10)

                shared_cross_entropy = tf.reduce_mean(-shared_onehot * tf.log(shared_predict + 1e-8))
                shared_train = tf.train.AdamOptimizer(0.00001).minimize(shared_cross_entropy)
        else:
            with tf.device('/cpu'):
                local_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
                local_conv1 = tf.layers.conv2d(
                    inputs=local_input, filters=32, kernel_size=[5, 5], strides=[2, 2],
                    padding='valid', activation=tf.nn.relu)
                local_conv2 = tf.layers.conv2d(
                    inputs=local_conv1, filters=32, kernel_size=[5, 5], strides=[2, 2],
                    padding='valid', activation=tf.nn.relu)
                local_flatten = tf.layers.flatten(local_conv2)
                local_predict = tf.layers.dense(inputs=local_flatten, units=10, activation=tf.nn.softmax)

                local_label = tf.placeholder(tf.int32, shape=[None])
                local_onehot = tf.one_hot(local_label, 10)

                local_cross_entropy = tf.reduce_mean(-local_onehot * tf.log(local_predict + 1e-8))
                local_train = tf.train.AdamOptimizer(0.00001).minimize(local_cross_entropy)


    session = tf.Session(server.target)
    session.run(tf.global_variables_initializer())

    ((train_data, train_labels),
        (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data = train_data/np.float32(255)
    train_data = np.expand_dims(train_data, axis=3)
    train_labels = train_labels.astype(np.int32)

    batch_size = 50
    data_size = len(train_data)
    epoch_size = int(data_size / batch_size)

    start_time = time.time()

    if is_learner:
        for i in range(epoch_size):
            batch_data = train_data[i*batch_size : (i+1)*batch_size]
            batch_label = train_labels[i*batch_size : (i+1)*batch_size]
            loss, _ = session.run(
                [shared_cross_entropy, shared_train],
                feed_dict={shared_input: batch_data, shared_label: batch_label})
            print(loss)
    
    else:
        while True:
            print('x')
        # for i in range(epoch_size):
        #     batch_data = train_data[i*batch_size : (i+1)*batch_size]
        #     batch_label = train_labels[i*batch_size : (i+1)*batch_size]
        #     loss, _ = session.run(
        #         [local_cross_entropy, local_train],
        #         feed_dict={local_input: batch_data, local_label: batch_label})
        #     print(loss)

    end_time = time.time()

    print(end_time - start_time)
        

if __name__ == '__main__':
    tf.app.run()

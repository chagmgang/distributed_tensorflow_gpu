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
                l2 = tf.layers.dense(inputs=l1, units=3, activation=tf.nn.softmax)

        ## build learner
        if is_learner:
            ph = tf.placeholder(tf.float32, shape=[None, 4])
            l1 = tf.layers.dense(inputs=ph, units=256, activation=tf.nn.relu)
            l2 = tf.layers.dense(inputs=l1, units=3, activation=tf.nn.softmax)

        sess = tf.Session(server.target)
        sess.run(tf.global_variables_initializer())

    if is_learner:
        while True:
            start_time = time.time()
            sess.run(
                l2,
                feed_dict={ph: [[1,2,3,4]]})
            end_time = time.time()

            print('gpu operation time : {}'.format(end_time - start_time))
            print('######')

    else:
        while True:
            start_time = time.time()
            sess.run(
                l2,
                feed_dict={ph: [[1,2,3,4]]})
            end_time = time.time()

            print('cpu operation time : {}'.format(end_time - start_time))
            print('######')

if __name__ == '__main__':
    tf.app.run()
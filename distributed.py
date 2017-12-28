# encoding:utf-8
import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags

flags.DEFINE_string('data_dir', '/home/zhangzhaoyu/incubator-mxnet-master/example/image-classification/data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 100000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

flags.DEFINE_string('ps_hosts', '172.16.1.182:2222', 'Comma-separated list of hostname:port pairs')

flags.DEFINE_string('worker_hosts', '172.16.1.183:2223,172.16.1.183:2224,172.16.1.187:2225,172.16.1.187:2226',
                    'Comma-separated list of hostname:port pairs')

flags.DEFINE_string('job_name', None, 'job name: worker or ps')

flags.DEFINE_integer('task_index', None, 'Index of task within the job')

flags.DEFINE_boolean("sync_replicas",False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are "
                     "aggregated before applied to avoid stale gradients")
flags.DEFINE_integer("replicas_to_aggregate",None,
                     "Number of replicas to aggregate before parameter "
                     "update is applied (For sync_replicas mode only; "
                     "default:num_workers)")

FLAGS = flags.FLAGS
IMAGE_PIXELS = 28

def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print 'job_name : %s' % FLAGS.job_name
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print 'task_index : %d' % FLAGS.task_index

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    worker_device = "/job:worker/task:%d/gpu:0" % FLAGS.task_index
    with tf.device(
        tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:ps/cpu:0",
            cluster=cluster)):
        global_step = tf.Variable(1, name='global_step', trainable=False)

        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name='hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                               stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
        sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        loss=tf.reduce_mean(loss)

        opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

        if FLAGS.sync_replicas:
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate=num_workers
            else:
                replicas_to_aggregate=FLAGS.replicas_to_aggregate

            opt=tf.train.SyncReplicasOptimizer(opt,
                                               replicas_to_aggregate=replicas_to_aggregate,
                                               total_num_replicas=num_workers,
                                               name='mnist_sync_replicas')

        train_opt = opt.minimize(loss, global_step=global_step)

        if FLAGS.sync_replicas and is_chief:
            chief_queue_runner=opt.get_chief_queue_runner()
            init_tokens_op=opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=train_dir, init_op=init_op,
                                 recovery_wait_secs=1,global_step=global_step)

        see_config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps",
                            "/job:worker/task:%d" % FLAGS.task_index])

        see_config.gpu_options.allow_growth=True

        if is_chief:
            print 'Worker %d: Initailizing session...' % FLAGS.task_index
        else:
            print 'Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index
        sess = sv.prepare_or_wait_for_session(server.target,config=see_config)
        print 'Worker %d: Session initialization  complete.' % FLAGS.task_index

        if FLAGS.sync_replicas and is_chief:
            print('Starting chief queue runner and running init_tokens_op')
            sv.start_queue_runners(sess,[chief_queue_runner])
            sess.run(init_tokens_op)

        time_begin = time.time()

        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}

            if(local_step%10000==0):
                validation_accuracy = accuracy.eval(feed_dict={
                    x: mnist.validation.images, y_: mnist.validation.labels},session=sess)
                print('Worker %d: validation accuracy %g' % (FLAGS.task_index,validation_accuracy))

            _,loss_value,step = sess.run([train_opt,loss,global_step], feed_dict=train_feed)
            local_step += 1

            train_accuracy = accuracy.eval(feed_dict={
                x: batch_xs, y_: batch_ys},session=sess)

            now = time.time()
            print 'Worker %d: traing step %d (global step:%d) loss %f training accuracy %g'  \
                  % (FLAGS.task_index, local_step, step,loss_value,train_accuracy)

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()

        train_time = time_end - time_begin
        print 'Training elapsed time:%f s' % train_time

        test_accuracy = accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels}, session=sess)
        print('Worker %d: test accuracy %g' % (FLAGS.task_index,test_accuracy))

if __name__ == '__main__':
    tf.app.run()
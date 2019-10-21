import argparse, time
# from yangNet import network
from PMVE_model import network
from UTILS_MF_ra import *

tf.logging.set_verbosity(tf.logging.WARN)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EXP_DATA = 'MF_qp37_ra_081001_PMVE'
LOW_DATA_PATH = r"D:\TJC\trainSet\ASCNNx2\qp37\low_data"
HIGH1_DATA_PATH = r"D:\TJC\trainSet\ASCNNx2\qp37\ASCNN_out_04"
HIGH2_DATA_PATH = r"D:\TJC\trainSet\ASCNNx2\qp37\ASCNN_out_13"

LABEL_PATH = r"D:\TJC\trainSet\ASCNNx2\qp37\label_s"
LOG_PATH = "./logs/%s/"%(EXP_DATA)
CKPT_PATH = "./checkpoints/%s/"%(EXP_DATA)
SAMPLE_PATH = "./samples/%s/"%(EXP_DATA)
PATCH_SIZE = (64, 64)
BATCH_SIZE = 64
BASE_LR = 3e-4
LR_DECAY_RATE = 0.5
LR_DECAY_STEP = 35
MAX_EPOCH = 2000


parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path

if __name__ == '__main__':

    #  return like this"[[[high1Data, lowData, high2Data], label], [[3, 8, 9], 33]]" with the whole path.
    train_list = get_train_list(load_file_list(HIGH1_DATA_PATH), load_file_list(LOW_DATA_PATH),
                                load_file_list(HIGH2_DATA_PATH), load_file_list(LABEL_PATH))

    with tf.name_scope('input_scope'):
        train_hight1Data = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_lowData = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_hight2Data = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
        train_gt = tf.placeholder('float32', shape=(BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    shared_model = tf.make_template('shared_model', network)
    train_output = shared_model(train_hight1Data, train_lowData, train_hight2Data)
    #train_output = shared_model(train_input)
    train_output = tf.clip_by_value(train_output, 0., 1.)
    with tf.name_scope('loss_scope'):
        loss2 = tf.reduce_sum(tf.square(tf.subtract(train_output, train_gt)))
        loss1 = tf.reduce_sum(tf.abs(tf.subtract(train_output, train_gt)))
        # for w in weights:
        W = tf.get_collection(tf.GraphKeys.WEIGHTS)
        for w in W:
            loss2 += tf.nn.l2_loss(w)*1e-4
        #tf.summary.scalar("loss", loss)

        avg_loss = tf.placeholder('float32')
        tf.summary.scalar("avg_loss", avg_loss)

    global_step     = tf.Variable(0, trainable=False) # len(train_list)
    learning_rate   = tf.train.exponential_decay(BASE_LR, global_step, LR_DECAY_STEP*1000, LR_DECAY_RATE, staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    # org ---------------------------------------------------------------------------------
    optimizer = tf.train.AdamOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(loss2, global_step=global_step)
    #
    # saver = tf.train.Saver(weights, max_to_keep=0)
    saver = tf.train.Saver(max_to_keep=0)
    # org end------------------------------------------------------------------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=config) as sess:
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        if not os.path.exists(os.path.dirname(CKPT_PATH)):
            os.makedirs(os.path.dirname(CKPT_PATH))
        if not os.path.exists(SAMPLE_PATH):
            os.makedirs(SAMPLE_PATH)

        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        sess.run(tf.global_variables_initializer())

        if model_path:
            print("restore model...")
            saver.restore(sess, model_path)
            print("Done")

        #for epoch in range(400, MAX_EPOCH):
        for epoch in range(MAX_EPOCH):
            total_g_loss, n_iter = 0, 0
            idxOfImgs = np.random.permutation(len(train_list))

            epoch_time = time.time()

            # for idx in range(len(idxOfImgs)):
            for idx in range(1000):
                input_high1Data, input_lowData, input_high2Data, gt_data = prepare_nn_data(train_list)
                feed_dict = {train_hight1Data: input_high1Data, train_lowData: input_lowData,
                             train_hight2Data: input_high2Data, train_gt: gt_data}
                _, l, output, g_step = sess.run([opt, loss2, train_output, global_step], feed_dict=feed_dict)
                total_g_loss += l
                n_iter += 1
                del input_high1Data, input_lowData, input_high2Data, gt_data, output
            lr, summary = sess.run([learning_rate, merged], {avg_loss:total_g_loss/n_iter})
            file_writer.add_summary(summary, epoch)
            print("Epoch: [%4d/%4d]  time: %4.4f\tloss: %.8f\tlr: %.8f"%(epoch, MAX_EPOCH, time.time()-epoch_time, total_g_loss/n_iter, lr))
            saver.save(sess, os.path.join(CKPT_PATH, "%s_%03d.ckpt"%(EXP_DATA, epoch)))

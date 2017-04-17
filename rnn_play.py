import tensorflow as tf
import numpy as np
import my_txtutils

ALPHA_SIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNAL_SIZE = 512

shakespearB10 = "tf_rnn_shakespearte_intermediate_checkpoints/rnn_test_minibatchseq_1477684737-47997000"
javaB0 = "checkpoints/rnn_train_1492352677-3000000"

coder = javaB0

ncnt = 0
with tf.Session() as sess:
    #new_saver = tf.train.import_meta_graph("tf_rnn_shakespearte_intermediate_checkpoints/rnn_test_minibatchseq_1477684737-47997000.meta");
    new_saver = tf.train.import_meta_graph("checkpoints/rnn_train_1492352677-3000000.meta")
    new_saver.restore(sess, coder)
    x = my_txtutils.convert_from_alphabet(ord("K"))
    x = np.array([[x]]) # [BATCH_SIZE, SEQLEN] with BATCH_SIZE = 1 and SEQLEN = 1

    # Initial value
    y = x
    h = np.zeros([1, INTERNAL_SIZE * NLAYERS], np.float32) # [BATCH_SIZE, INTERNAL_SIZE * SEQLEN]
    file = open("CodedByRnn_topn10.java", "w")

    for i in range (1000000000):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0':1})
        c = my_txtutils.sample_from_probabilities(yo, topn=10)
        y = np.array([[c]]) # [BATCH_SIZE, SEQLEN] with BATCH_SIZE = 1 and SEQLEN = 1
        c = chr(my_txtutils.convert_to_alphabet(c))
        file.write(c)
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1

        if ncnt == 100:
            print("")
            ncnt = 0


    file.close()
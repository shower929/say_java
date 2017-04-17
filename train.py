import tensorflow as tf
import my_txtutils as txt
import time
import math
import os
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

tf.set_random_seed(0)

SEQ_LENGTH = 30
BATCH_SIZE = 100
ALPHA_SIZE = txt.ALPHASIZE
INTERNAL_SIZE = 512
NLAYERS = 3

learning_rate = 0.001
dropout_pkeep = 0.5 # No dropout

# Load training data
# Shakespeare
# train_data = "shakespeare/*.txt"

# Java code
train_data = "../Desktop/android/frameworks/**/*.java"

codetxt, valitext, bookranges = txt.read_data_files(train_data, validation=True)

# Model
lr = tf.placeholder(tf.float32, name='lr')
pkeep = tf.placeholder(tf.float32, name='pkeep')
batchsize = tf.placeholder(tf.int32, name='batchsize')

# Input
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [BATCH_SIZE, SEQ_LEN]
Xo = tf.one_hot(X, ALPHA_SIZE, 1.0, 0.0)    # [BATCH_SIZE, SEQ_LEN, ALPHA_SIZE]

# Output
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [BATCH_SIZE, SEQ_LEN]
Yo_ = tf.one_hot(Y_, ALPHA_SIZE, 1.0, 0.0)  # [BATCH_SIZE, SEQ_LEN, ALPHA_SIZE]

# Input state
Hin = tf.placeholder(tf.float32, [None, INTERNAL_SIZE * NLAYERS], name='Hin')   # [BATCH_SIZE, INTERNAL_SIZE * NLAYERS]

# GRU sell
onecell = rnn.GRUCell(INTERNAL_SIZE)
# Apply dropout to GRU cell
dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
# Stack GRU cells
multicell = rnn.MultiRNNCell([dropcell] * NLAYERS, state_is_tuple=False)
# Apply dropout to GRU cell stack
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
epoch_size = len(codetxt) // BATCH_SIZE * SEQ_LENGTH
txt.print_data_stats(len(codetxt), len(valitext), epoch_size)

# Input x and output Y and H (internal state)
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [BATCHSIZE, SEQLEN, INTERNAL_SIZE]
# H: [BATCHSIZE, INTERNAL_SIZE * NLAYERS]

H = tf.identity(H, name='H') # Just to give it a name

Yflat = tf.reshape(Yr, [-1, INTERNAL_SIZE]) # [BATCH_SIZE * SEQ_LEN, INTERNAL_SIZE]
# Sum of layer for each alpha
Ylogits = layers.linear(Yflat, ALPHA_SIZE) # [BATCH_SIZE * SEQ_LEN, ALPHA_SIZE]
Yflat_ = tf.reshape(Yo_, [-1, ALPHA_SIZE]) # [BATCH_SIZE * SEQ_LEN, ALPHA_SIZE]
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)   # [BATCH_SIZE * SEQ_LEN]
loss = tf.reshape(loss, [batchsize, -1])    # [BATCH_SIZE, SEQ_LEN]
Yo = tf.nn.softmax(Ylogits, name='Yo')  # [BATCH_SIZE * SEQ_LEN, ALPHA_SIZE]
Y = tf.argmax(Yo, 1)    # [BATCH_SIZE * SEQ_LEN]
Y = tf.reshape(Y, [batchsize, -1], name='Y') # [BATCH_SIZE, SEQ_LEN]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

timestamp = str(math.trunc(time.time()))

training_writter = tf.summary.FileWriter("java-log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("java-log/" + timestamp + "-validation")

if not os.path.exists("java-checkpoints"):
    os.mkdir("java-checkpoints")
saver = tf.train.Saver(max_to_keep=1)

DISPLAY_FREQ = 50
_50_BATCHS = DISPLAY_FREQ * BATCH_SIZE * SEQ_LENGTH
progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next " + str(DISPLAY_FREQ) + "batches")

# Init
istate = np.zeros([BATCH_SIZE, INTERNAL_SIZE * NLAYERS])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

for x, y_, epoch in txt.rnn_minibatch_sequencer(codetxt, BATCH_SIZE, SEQ_LENGTH, nb_epochs=1000):
    feed_dict = {X:x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCH_SIZE}
    _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

    # Save training data for Tensorboard
    training_writter.add_summary(smm, step)

    if step % _50_BATCHS == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCH_SIZE} # No dropout for validation
        y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)

    if step % _50_BATCHS == 0 and len(valitext) > 0:
        VALI_SEQLEN = 1*1024
        bsize = len(valitext) // VALI_SEQLEN
        txt.print_validation_header(len(codetxt), bookranges)
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1)) # All data in 1 batch
        vali_nullstate = np.zeros([bsize, INTERNAL_SIZE * NLAYERS])
        feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0, batchsize: bsize} # No dropout for variation
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_validation_stats(ls, acc)
        validation_writer.add_summary(smm, step)

    # Display a short text generated with the current weights and biases (every 150 batches)
    if step // 3 % _50_BATCHS == 0:
        txt.print_text_generation_header()
        ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
        rh = np.zeros([1, INTERNAL_SIZE * NLAYERS])
        for k in range(1000):
            ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
            rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
            print(chr(txt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
        txt.print_text_generation_footer()

    if step // 10 % _50_BATCHS == 0:
        saver.save(sess, 'java-checkpoints/rnn_train_' + timestamp, global_step=step)

    # Display progress bar
    progress.step(reset=step % _50_BATCHS == 0)

    # Loop state around
    istate = ostate
    step += BATCH_SIZE * SEQ_LENGTH


import tensorflow as tf
import time
from sklearn.metrics import roc_curve, auc  # curve曲线
import os
from util import *

def Model(i, j, lr, ds, dr, t_step, batch_size, py_title, title, sess, AP_dataTrain,
                    AP_labelTrain, AP_dataTest, AP_labelTest,PVP_dataTrain, PVP_labelTrain,
                    PVP_dataTest, PVP_labelTest,PC_dataTrain, PC_labelTrain,PC_dataTest, PC_labelTest):
    global_step = tf.Variable(0)
    keep_prob = tf.placeholder(tf.float32)

    x1 = tf.placeholder(tf.float32, [None, 16*16*16 * 1])
    label1 = tf.placeholder(tf.float32, [None, 2])

    inputData_1 = tf.reshape(x1, [-1, 16,16,16, 1])

    kernel_11 = weight_variable([3, 3,3, 1, 32])
    bias_11 = bias_variable([32])
    conv_11 = conv3d(inputData_1, kernel_11)
    conv_out_11 = tf.nn.relu(conv_11 + bias_11)
    pooling_out_11 = max_pool_2x2_3D(conv_out_11)

    kernel_12 = weight_variable([3, 3,3, 32, 64])
    bias_12 = bias_variable([64])
    conv_12 = conv3d(pooling_out_11, kernel_12)
    conv_out_12 = tf.nn.relu(conv_12 + bias_12)
    pooling_out_12 = max_pool_2x2_3D(conv_out_12)

    pooling_out_12 = tf.reshape(pooling_out_12, [-1, 7 * 7 * 64])

    w_fc_11 = weight_variable([7 * 7 * 64, 500])
    b_fc_11 = bias_variable([500])
    fc_out_11 = tf.nn.relu(tf.matmul(pooling_out_12, w_fc_11) + b_fc_11)
    drop11 = tf.nn.dropout(fc_out_11, keep_prob)

    w_fc_12 = weight_variable([500, 50])
    b_fc_12 = bias_variable([50])
    fc_out_12 = tf.nn.relu(tf.matmul(drop11, w_fc_12) + b_fc_12)

    ######################################   Coronal_Feature   ###################################

    x2 = tf.placeholder(tf.float32, [None, 16*16*16 * 1])
    label2 = tf.placeholder(tf.float32, [None, 2])

    inputData_2 = tf.reshape(x2, [-1, 16,16,16, 1])

    kernel_21 = weight_variable([3, 3,3, 1, 32])
    bias_21 = bias_variable([32])
    conv_21 = conv3d(inputData_2, kernel_21)
    conv_out_21 = tf.nn.relu(conv_21 + bias_21)

    pooling_out_21 = max_pool_2x2_3D(conv_out_21)

    kernel_22 = weight_variable([3, 3,3, 32, 64])
    bias_22 = bias_variable([64])
    conv_22 = conv3d(pooling_out_21, kernel_22)
    conv_out_22 = tf.nn.relu(conv_22 + bias_22)
    pooling_out_22 = max_pool_2x2_3D(conv_out_22)

    pooling_out_22 = tf.reshape(pooling_out_22, [-1, 7 * 7 * 64])

    w_fc_21 = weight_variable([7 * 7 * 64, 500])
    b_fc_21 = bias_variable([500])
    fc_out_21 = tf.nn.relu(tf.matmul(pooling_out_22, w_fc_21) + b_fc_21)
    drop21 = tf.nn.dropout(fc_out_21, keep_prob)

    w_fc_22 = weight_variable([500, 50])
    b_fc_22 = bias_variable([50])
    fc_out_22 = tf.nn.relu(tf.matmul(drop21, w_fc_22) + b_fc_22)

    ######################################   Sagittal_Feature   ###################################

    x3 = tf.placeholder(tf.float32, [None, 16*16*16 * 1])
    label3 = tf.placeholder(tf.float32, [None, 2])

    inputData_3 = tf.reshape(x3, [-1, 16,16,16, 1])

    kernel_31 = weight_variable([3, 3,3, 1, 32])
    bias_31 = bias_variable([32])
    conv_31 = conv3d(inputData_3, kernel_31)
    conv_out_31 = tf.nn.relu(conv_31 + bias_31)
    pooling_out_31 = max_pool_2x2_3D(conv_out_31)

    kernel_32 = weight_variable([3, 3,3, 32, 64])
    bias_32 = bias_variable([64])
    conv_32 = conv3d(pooling_out_31, kernel_32)
    conv_out_32 = tf.nn.relu(conv_32 + bias_32)
    pooling_out_32 = max_pool_2x2_3D(conv_out_32)

    pooling_out_32 = tf.reshape(pooling_out_32, [-1, 7 * 7 * 64])

    w_fc_31 = weight_variable([7 * 7 * 64, 500])
    b_fc_31 = bias_variable([500])
    fc_out_31 = tf.nn.relu(tf.matmul(pooling_out_32, w_fc_31) + b_fc_31)
    drop31 = tf.nn.dropout(fc_out_31, keep_prob)

    w_fc_32 = weight_variable([500, 50])
    b_fc_32 = bias_variable([50])
    fc_out_32 = tf.nn.relu(tf.matmul(drop31, w_fc_32) + b_fc_32)
    
    
    ###################   Common_Indevidual Analysis for XOY_Frature   ########################
    common_V1 = weight_variable([50, 50])
    bias_common1 = bias_variable([50])
    common_feature1 = tf.nn.relu(tf.matmul(fc_out_12, common_V1) + bias_common1)

    indevidal_Q1 = tf.placeholder(tf.float32, [50, 50])
    bias_indevidual1 = bias_variable([50])
    indevidual_feature1 = tf.nn.relu(tf.matmul(fc_out_12, indevidal_Q1) + bias_indevidual1)

    ###################   Common_Indevidual Analysis for XOZ_Frature   ########################
    common_V2 = weight_variable([50, 50])
    bias_common2 = bias_variable([50])
    common_feature2 = tf.nn.relu(tf.matmul(fc_out_22, common_V2) + bias_common2)

    indevidal_Q2 = tf.placeholder(tf.float32, [50, 50])
    bias_indevidual2 = bias_variable([50])
    indevidual_feature2 = tf.nn.relu(tf.matmul(fc_out_22, indevidal_Q2) + bias_indevidual2)

    ###################   Common_Indevidual Analysis for XOZ_Frature   ########################
    common_V3 = weight_variable([50, 50])
    bias_common3 = bias_variable([50])
    common_feature3 = tf.nn.relu(tf.matmul(fc_out_32, common_V3) + bias_common3)

    indevidal_Q3 = tf.placeholder(tf.float32, [50, 50])
    bias_indevidual3 = bias_variable([50])
    indevidual_feature3 = tf.nn.relu(tf.matmul(fc_out_32, indevidal_Q3) + bias_indevidual3)

    #####################################     Fusion      ####################################
    feature_cat = tf.concat(
        [common_feature1, common_feature2, common_feature3, indevidual_feature1, indevidual_feature2, indevidual_feature3],
        1)

    w_fc_f1 = weight_variable([50*6,50])
    b_fc_f1 = bias_variable([50])
    fc_f_out1 = tf.nn.relu(tf.matmul(feature_cat, w_fc_f1) + b_fc_f1)


    w_fc_f2 = weight_variable([50, 2])
    b_fc_f2 = bias_variable([2])
    mid = tf.matmul(fc_f_out1, w_fc_f2) + b_fc_f2

    prediction = tf.nn.softmax(mid)

    ################################################
    with tf.name_scope('loss'):
        CL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mid, labels=label1))
        tf.summary.scalar('CL', CL)

        fai = 0.5 / 14406
        thta = 0.5 / 1029

        part1_1 = tf.norm((common_feature1 - common_feature2), 'fro', axis=(0, 1))
        part1_2 = tf.norm((common_feature1 - common_feature3), 'fro', axis=(0, 1))
        part1_3 = tf.norm((common_feature2 - common_feature3), 'fro', axis=(0, 1))

        part2_1 = tf.norm((fc_out_12 - tf.matmul(fc_out_12, tf.matmul(tf.transpose(common_V1), common_V1)) - tf.matmul(
            fc_out_12, tf.matmul(tf.transpose(indevidal_Q1), indevidal_Q1))), 'fro', axis=(0, 1))
        part2_2 = tf.norm((fc_out_22 - tf.matmul(fc_out_22, tf.matmul(tf.transpose(common_V2), common_V2)) - tf.matmul(
            fc_out_22, tf.matmul(tf.transpose(indevidal_Q2), indevidal_Q2))), 'fro', axis=(0, 1))
        part2_3 = tf.norm((fc_out_32 - tf.matmul(fc_out_32, tf.matmul(tf.transpose(common_V3), common_V3)) - tf.matmul(
            fc_out_32, tf.matmul(tf.transpose(indevidal_Q3), indevidal_Q3))), 'fro', axis=(0, 1))

        part3_1 = tf.norm((tf.matmul(tf.transpose(common_V1), indevidal_Q1)), 'fro', axis=(0, 1))
        part3_2 = tf.norm((tf.matmul(tf.transpose(common_V2), indevidal_Q2)), 'fro', axis=(0, 1))
        part3_3 = tf.norm((tf.matmul(tf.transpose(common_V3), indevidal_Q3)), 'fro', axis=(0, 1))

        loss_CI = (part1_1 + part1_2 + part1_3) + fai * (part2_1 + part2_2 + part2_3) + thta * (part3_1 + part3_2 + part3_3)

        total_loss = CL + loss_CI
        tf.summary.scalar('total_loss', total_loss)

    learning_rate = tf.train.exponential_decay(lr, global_step, decay_stePC=ds, decay_rate=dr,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    with tf.name_scope('Accuracy'):
        output_position = tf.argmax(prediction, 1)
        label_position = tf.argmax(label1, 1)
        predict = tf.equal(output_position, label_position)
        Accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))
        tf.summary.scalar('Accuracy', Accuracy)

    merge = tf.summary.merge_all()
    #####################################    Train    ##########################################

    sess.run(tf.global_variables_initializer())
    tf.reset_default_graph()

    board_path = "/home/public/RCICA/log/" + py_title
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + title
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + i
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    board_path = board_path + '/' + j
    if (not (os.path.exists(board_path))):
        os.mkdir(board_path)

    test_board_path = board_path + '/' + 'test'
    if (not (os.path.exists(test_board_path))):
        os.mkdir(test_board_path)

    train_board_path = board_path + '/' + 'train'
    if (not (os.path.exists(train_board_path))):
        os.mkdir(train_board_path)

    test_writer = tf.summary.FileWriter(test_board_path + '/', tf.get_default_graph())
    train_writer = tf.summary.FileWriter(train_board_path + '/', tf.get_default_graph())

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)

    before = time.time()
    trigger=1024*1024
    flag=0
    early_stopping=100

    for times in range(t_step):
        AP_dataTest_r, AP_labelTest_r, PVP_dataTest_r, PVP_labelTest_r, \
        PC_dataTest_r, PC_labelTest_r, = sess.run(
            [AP_dataTest, AP_labelTest, PVP_dataTest, PVP_labelTest, PC_dataTest, PC_labelTest])
        AP_dataTrain_r, AP_labelTrain_r, PVP_dataTrain_r, PVP_labelTrain_r, \
        PC_dataTrain_r, PC_labelTrain_r = sess.run(
            [AP_dataTrain, AP_labelTrain, PVP_dataTrain, PVP_labelTrain, PC_dataTrain,
             PC_labelTrain])

        ###########################  test  #######################
        if times % 10 == 0:
            total_loss_r, summary, acc, output_position_r, label_position_r, \
            predict_r, p, loss_cross_entropy_r = sess.run(
                [total_loss, merge, Accuracy, output_position,
                 label_position, predict, prediction, CL],
                feed_dict={x1: AP_dataTest_r, label1: AP_labelTest_r, x2: PVP_dataTest_r,
                           label2: PVP_labelTest_r, x3: PC_dataTest_r, label3: PC_labelTest_r,
                           keep_prob: 1.0})
            test_writer.add_summary(summary, times)
            sen, spe = Sensitivity_specificity(output_position_r, predict_r)
            fpr, tpr, thresholds = roc_curve(label_position_r, p[:, 1], drop_intermediate=False)
            AUC = auc(fpr, tpr)

            if trigger<total_loss_r:
                trigger=total_loss_r
                flag=0

            if flag==early_stopping:
                break
        ###########################  train  #######################
        if times % 99 == 0:
            summary, _ = sess.run([merge, train_step],
                                  feed_dict={x1: AP_dataTrain_r, label1: AP_labelTrain_r,
                                             x2: PVP_dataTrain_r, label2: PVP_labelTrain_r,
                                             x3: PC_dataTrain_r, label3: PC_labelTrain_r,
                                             keep_prob: 0.5})
            train_writer.add_summary(summary, times)
        else:
            sess.run([train_step],
                     feed_dict={x1: AP_dataTrain_r, label1: AP_labelTrain_r, x2: PVP_dataTrain_r,
                                label2: PVP_labelTrain_r, x3: PC_dataTrain_r,
                                label3: PC_labelTrain_r, keep_prob: 0.5})


    after = time.time()
    train_writer.close()
    test_writer.close()

    return acc, sen, spe, AUC, output_position_r, label_position_r, predict_r, p, after, before
import time
import itertools
from PMVE_model import network
from PSVE_model import model_single
# ASCNN(0,4)(1,3)
from UTILS_MF_ra import *

tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EXP_DATA1 = "MF_qp37_PMVE_999"  # double
EXP_DATA2 = "qp37_ra_PSVE_999"  # 暂时不用！
MODEL_DOUBLE_PATH1 = "./checkpoints/%s/"%(EXP_DATA1)
MODEL_DOUBLE_PATH2 = "./checkpoints/%s/"%(EXP_DATA2)
#ORIGINAL_PATH = "./data/test/mix_noSAO/test_D/q22"
#GT_PATH = "./data/test/mix/test_D"
# path 1 : 04
# path 2 : 13
# QP : "rec dir"
# GT_path : "org"

# 139 line.
ASCNN_Frames_PATH1 = r"I:\TJC\ldp_new\ASCNN_out\yuv_04\D\RaceHorses_qp22_ai_416x240"
ASCNN_Frames_PATH2 = r"I:\TJC\ldp_new\ASCNN_out\yuv_13\D\RaceHorses_qp22_ai_416x240"
QP_LOWDATA_PATH = r'I:\TJC\ldp_new\REC\D\RaceHorses_qp37_ldp_416x240'
GT_PATH = r"I:\TJC\HEVC_TestSequenct\org\D\RaceHorses_416x240"
DL_path = r'I:\TJC\HEVC_TestSequenct\DL_rec\abc'
OUT_DATA_PATH = "./outdata/%s/"%(EXP_DATA1)
NOFILTER = {'q22':42.2758, 'q27':38.9788, 'qp32':35.8667, 'q37':32.8257,'qp37':32.8257}

# Ground truth images dir should be the 2nd component of 'fileOrDir' if 2 components are given.

##  cb, cr components are not implemented
# [ASCNN_Frames_PATH, QP_LOWDATA_PATH, GT_PATH]
def prepare_test_data(fileOrDir):
    doubleData_ycbcr = []
    doubleGT_y = []
    singleData_ycbcr = []
    singleGT_y = []

    fileName_list = []
    #The input is a single file.
    if len(fileOrDir) == 4:
        # return the whole absolute path.
        fileName_list = load_file_list(fileOrDir[1])
        # [[ASCNN_Frame, lowData, LabelData],[ASCNN_Frame2, lowData2, LabelData2], []]
        # single_list # [[low1, lable1], [2,2] ....]
        double_list, single_list = get_test_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]),
                                   load_file_list(fileOrDir[2]), load_file_list(fileOrDir[3]))

        for pair in double_list:
            ASCNN_Data_List1 = []
            lowData_List = []
            ASCNN_Data_List2 = []

            ASCNN_Data_imgY1 = c_getYdata(pair[0])
            lowData_imgY = c_getYdata(pair[1])
            ASCNN_Data_imgY2 = c_getYdata(pair[2])
            CbCr = c_getCbCr(pair[1])
            gt_imgY = c_getYdata(pair[3])

            #normalize
            ASCNN_Data_imgY1 = normalize(ASCNN_Data_imgY1)
            lowData_imgY = normalize(lowData_imgY)
            ASCNN_Data_imgY2 = normalize(ASCNN_Data_imgY2)

            ASCNN_Data_imgY1 = np.resize(ASCNN_Data_imgY1, (1, ASCNN_Data_imgY1.shape[0], ASCNN_Data_imgY1.shape[1],1))
            lowData_imgY = np.resize(lowData_imgY, (1, lowData_imgY.shape[0], lowData_imgY.shape[1], 1))
            ASCNN_Data_imgY2 = np.resize(ASCNN_Data_imgY2, (1, ASCNN_Data_imgY2.shape[0], ASCNN_Data_imgY2.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1],1))

            ## act as a placeholder

            ASCNN_Data_List1.append([ASCNN_Data_imgY1, 0])
            lowData_List.append([lowData_imgY, CbCr])
            ASCNN_Data_List2.append([ASCNN_Data_imgY2, 0])

            doubleData_ycbcr.append([ASCNN_Data_List1, lowData_List, ASCNN_Data_List2])
            # [ASCNN_Data_imgY, 0]
            doubleGT_y.append(gt_imgY)

        # single_list # [[low1, lable1], [2,2] ....]
        for pair in single_list:
            lowData_list = []
            lowData_imgY = c_getYdata(pair[0])
            CbCr = c_getCbCr(pair[0])
            gt_imgY = c_getYdata(pair[1])

            # normalize
            lowData_imgY = normalize(lowData_imgY)

            lowData_imgY = np.resize(lowData_imgY, (1, lowData_imgY.shape[0], lowData_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            lowData_list.append([lowData_imgY, CbCr])
            singleData_ycbcr.append(lowData_list)
            singleGT_y.append(gt_imgY)

    else:
        print("Invalid Inputs...!tjc!")
        exit(0)

    return doubleData_ycbcr, doubleGT_y, singleData_ycbcr, singleGT_y, fileName_list


def abc(modelPath1, modelPath2, fileOrDir):
    max = [0, 0]

    tem1 = [f for f in os.listdir(modelPath1) if 'data' in f]
    ckptFiles1 = sorted([r.split('.data')[0] for r in tem1])

    tem2 = [f for f in os.listdir(modelPath2) if 'data' in f]
    ckptFiles2 = sorted([r.split('.data')[0] for r in tem2])

    re_psnr = tf.placeholder('float32')
    tf.summary.scalar('re_psnr', re_psnr)


    doubleData_ycbcr, doubleGT_y, singleData_ycbcr, singleGT_y, fileName_list = prepare_test_data(fileOrDir)
    total_time, total_psnr = 0, 0
    total_psnr_36 = 0
    total_imgs = len(fileName_list)
    total_hevc_psnr = 0
    total_hevc_psnr_36 = 0
    count = 0
    for i in range(total_imgs):
        # print(fileName_list[i])
        if i % 4 != 0:
            count += 1

            # sorry! this place write so difficult!【[[[h1,0]],[[low,0]],[[h2, 0]]], [[[h1,0]],[[low,0]],[[h2, 0]]]】
            j = i - (i//4) - 1
            ASCNNDataY1 = doubleData_ycbcr[j][0][0][0]
            # print(np.shape(ASCNNDataY1))

            imgLowDataY = doubleData_ycbcr[j][1][0][0]
            imgLowCbCr = doubleData_ycbcr[j][1][0][1]
            ASCNNDataY2 = doubleData_ycbcr[j][2][0][0]

            #imgCbCr = original_ycbcr[i][1]
            gtY = doubleGT_y[j] if doubleGT_y else 0

            for ckpt1 in ckptFiles1:
                epoch = int(ckpt1.split('_')[-1].split('.')[0])
                if epoch != 999:
                    continue
                # very important!!!!!!!
                tf.reset_default_graph()
                # Double section
                ASCNN_Data_tensor1 = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                lowData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                ASCNN_Data_tensor2 = tf.placeholder(tf.float32, shape=(1, None, None, 1))

                shared_model1 = tf.make_template('shared_model', network)
                output_tensor1 = shared_model1(ASCNN_Data_tensor1, lowData_tensor, ASCNN_Data_tensor2)
                output_tensor1 = tf.clip_by_value(output_tensor1, 0., 1.)
                output_tensor1 = output_tensor1 * 255
                with tf.Session() as sess:
                    # saver = tf.train.Saver(tf.global_variables())
                    saver = tf.train.Saver()
                    sess.run(tf.global_variables_initializer())

                    saver.restore(sess, os.path.join(modelPath1, ckpt1))

                    ###tjc###
                    start_t = time.time()
                    out = sess.run(output_tensor1, feed_dict={
                        ASCNN_Data_tensor1: ASCNNDataY1, lowData_tensor:imgLowDataY, ASCNN_Data_tensor2: ASCNNDataY2})
                    duration_t = time.time() - start_t
                    total_time += duration_t
                    hevc = psnr(imgLowDataY * 255.0, gtY)
                    print("hevc:",hevc)
                    total_hevc_psnr += hevc
                    total_hevc_psnr_36 += hevc


                    out = np.around(out)
                    out = out.astype('int')
                    out = np.reshape(out, [1, out.shape[1], out.shape[2], 1])

                    Y = np.reshape(out, [out.shape[1], out.shape[2]])
                    Y = np.array(list(itertools.chain.from_iterable(Y)))
                    U = imgLowCbCr[0]
                    V = imgLowCbCr[1]
                    creatPath = os.path.join(DL_path, fileName_list[i].split('\\')[-2])
                    if not os.path.exists(creatPath):
                        os.mkdir(creatPath)

                    # print(np.shape(gtY))

                    if doubleGT_y:
                        p = psnr(out, gtY)
                        print("p:",p)

                        path = os.path.join(DL_path,
                                            fileName_list[i].split('\\')[-2],
                                            fileName_list[i].split('\\')[-1].split('.')[0]) + '_%.4f' % (p-hevc)+ '.yuv'
                        # print(path)
                        YUV = np.concatenate((Y, U, V))
                        print(np.shape(Y))
                        print(np.shape(U))
                        print(np.shape(V))
                        YUV = YUV.astype('uint8')
                        YUV.tofile(path)

                        total_psnr += p
                        total_psnr_36 += p
                        print("qp37\tepoch:%d\t%s\t%.4f\n" % (epoch, fileName_list[i], p))

        else:   ##single frame
            continue
            count += 1
            j = i // 4
            ## ???
            lowDataY = singleData_ycbcr[j][0][0]
            imgLowCbCr = singleData_ycbcr[j][0][1]
            # imgCbCr = original_ycbcr[i][1]
            gtY = singleGT_y[j] if singleGT_y else 0

            hevc = psnr(lowDataY * 255.0, gtY)
            total_hevc_psnr += hevc
            print(hevc)

            #start_t = time.time()
            for ckpt2 in ckptFiles2:
                epoch = int(ckpt2.split('_')[-1].split('.')[0])
                if epoch != 999:
                    continue

                tf.reset_default_graph()
                # Single section
                lowSingleData_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
                shared_model2 = tf.make_template('shared_model', model_single)
                output_tensor2 = shared_model2(lowSingleData_tensor)
                # output_tensor = shared_model(input_tensor)
                output_tensor2 = tf.clip_by_value(output_tensor2, 0., 1.)
                output_tensor2 = output_tensor2 * 255
                with tf.Session() as sess:
                    saver = tf.train.Saver(tf.global_variables())
                    sess.run(tf.global_variables_initializer())

                    saver.restore(sess, os.path.join(modelPath2, ckpt2))
                    out = sess.run(output_tensor2, feed_dict={lowSingleData_tensor: lowDataY})
                    out = np.around(out)
                    out = out.astype('int')
                    out = np.reshape(out, [1, out.shape[1], out.shape[2], 1])
                    Y = np.reshape(out, [out.shape[1], out.shape[2]])
                    Y = np.array(list(itertools.chain.from_iterable(Y)))
                    U = imgLowCbCr[0]
                    V = imgLowCbCr[1]
                    creatPath = os.path.join(DL_path, fileName_list[i].split('\\')[-2])
                    if not os.path.exists(creatPath):
                        os.mkdir(creatPath)


                    if singleGT_y:
                        p = psnr(out, gtY)

                        path = os.path.join(DL_path, fileName_list[i].split('\\')[-2],
                                            fileName_list[i].split('\\')[-1].split('.')[0]) + '_%.4f' % (p - hevc) + '.yuv'

                        YUV = np.concatenate((Y, U, V))
                        YUV = YUV.astype('uint8')
                        YUV.tofile(path)


                        total_psnr += p
                        print("qp37\tepoch:%d\t%s\t%.4f\n" % (epoch, fileName_list[i], p))

            #duration_t = time.time() - start_t
            #total_time += duration_t


    print("AVG_DURATION:%.2f\tAVG_PSNR:%.4f"%(total_time/total_imgs, total_psnr / count))
    print('Total_time:', total_time)
    print('count:', count)
    print("%.4f"%(total_hevc_psnr/count))
    print("%.4f" % (total_hevc_psnr_36 / 36))
    print("%.4f" % (total_psnr_36 / 36))
    # avg_psnr = total_psnr/total_imgs
    avg_psnr = total_psnr / count
    avg_duration = (total_time/total_imgs)
    if avg_psnr > max[0]:
        max[0] = avg_psnr
        max[1] = epoch


        # summary = sess.run(merged, {re_psnr:avg_psnr})
        # file_writer.add_summary(summary, epoch)
        # tf.logging.warning("AVG_DURATION:%.2f\tAVG_PSNR:%.2f\tepoch:%d"%(avg_duration, avg_psnr, epoch))

    # QP = os.path.basename(HIGHDATA_PATH)
    # tf.logging.warning("QP:%s\tepoch: %d\tavg_max:%.4f\tdelta:%.4f"%(QP, max[1], max[0], max[0]-NOFILTER[QP]))


if __name__ == '__main__':
    abc(MODEL_DOUBLE_PATH1, MODEL_DOUBLE_PATH2, [ASCNN_Frames_PATH1, QP_LOWDATA_PATH, ASCNN_Frames_PATH2, GT_PATH])
    # test_all_ckpt(MODEL_PATH, [r'D:\PycharmProjects\data_tjc\hm_test_noFilter\qp37\data', r'D:\PycharmProjects\data_tjc\hm_test_origin\org'])

print('tjc')
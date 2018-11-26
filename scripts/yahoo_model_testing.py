from subprocess import call
import numpy as np
import os
import caffe
import shutil
from keras.models import load_model
from PIL import  Image


def violence_adult_scores(vid_id_list):
    if len(vid_id_list) == 0:
        print("video id's list is empty")
    curr_dir = os.getcwd()
    nsfw_net_yahoo = caffe.Net('/home/ubuntu/ML/user/Minhaj/yahoo_finetune/files/model/deploy_yahoo_test.prototxt',
                                '/home/ubuntu/ML/user/Minhaj/yahoo_finetune/models/caffe_model_2/caffe_model_2_iter_3000.caffemodel',
                                caffe.TEST)
    # nsfw_net_2k = caffe.Net(curr_dir + '/adult_model/deploy_p_2k.prototxt',
    #                        curr_dir '/adult_model/_iter_20000.caffemodel', caffe.TEST)
    # nsfw_net_2k = caffe.Net('/home/ubuntu/ML/deep_learning/projects/public_github/adult_content_detection/yahoo_open_nsfw/trained_models/v3/deploy.prototxt',
    #                                                '/home/ubuntu/ML/deep_learning/projects/public_github/adult_content_detection/yahoo_open_nsfw/trained_models/v3/_iter_2000.caffemodel', caffe.TEST)
    #nsfw_net_2k = caffe.Net(
    #    '/home/ubuntu/ML/deep_learning/projects/public_github/adult_content_detection/yahoo_open_nsfw/trained_models/v3/deploy.prototxt',
    #    '/home/ubuntu/ML/deep_learning/projects/public_github/adult_content_detection/yahoo_open_nsfw/open_nsfw/nsfw_model/_iter_1000_lr_0.001.caffemodel',
    #    caffe.TEST)
    # nsfw_net_deep_miles = caffe.Net(
    #     '/home/ubuntu/ML/deep_learning/projects/public_github/adult_content_detection/yahoo_open_nsfw/trained_models/v3/deploy.prototxt',
    #     '/home/ubuntu/ML/deep_learning/caffe/caffe-rc5/models/bvlc_alexnet/adult_deep_miles_v6.2_lr_0.1/_iter_1000.caffemodel',
    #     caffe.TEST)
    # nsfw_net_2k = caffe.Net(
    #     '/home/ubuntu/ML/deep_learning/projects/public_github/adult_content_detection/yahoo_open_nsfw/trained_models/v3/deploy.prototxt',
    #     # '/home/ubuntu/ML/deep_learning/caffe/caffe-rc5/models/bvlc_alexnet/adult_v6_lr_0.1/_iter_2000.caffemodel' ,
    #     '/home/ubuntu/ML/deep_learning/caffe/caffe-rc5/models/bvlc_alexnet/adult_v6.2_lr_0.1/_iter_1000.caffemodel' ,
    #     caffe.TEST)
    # #adult_v6_lr_0.01/iter_2000.caffemodel'adult_v6_lr_0.1/_iter_500.caffemodel'
    # # adult_vid_scores = open("adult_scores.txt", "w")
    # inc_res = '/home/ubuntu/ML/deep_learning/projects/vidooly/vidooly/machinelearning/brand_safety/models/adult/inception_resnet_v2_adult_v2_final_8_sep.h5'
    # inc_res = "../models/v3/model_weights.h5"
    # inc_res = "/home/vidooly/DeepLearning/ml/projects/user/minhaj/caffe_test/models/inception_resnet_v2_adult_v2_final_8_sep.h5"
    # model_incep = load_model(inc_res)
    # model_incep.summary()
    # exit()

    for vid_id in vid_id_list:
        '''
        vid_url = 'www.youtube.com/watch?v=' + vid_id
        out_dir_videos = curr_dir + '/downloaded_videos/'
        out_dir_frames = curr_dir + '/video_frames/' + vid_id
        #out_dir_frames = '/home/ubuntu/ML/deep_learning/projects/public_github/adult_content_detection/yahoo_open_nsfw/data/adult/failed_adult/additional_data/train_frames_removed'
        command = "youtube-dl -f 'bestvideo[height<=720]' -o " + out_dir_videos + "%s.mp4 %s" % (vid_id, vid_url)
        print(command)
        call(command, shell=True)
        
        if not os.path.exists(out_dir_frames):
            os.makedirs(out_dir_frames)
            

        command_2 = "ffmpeg -i {0} -vf fps=1 {1}img%03d.jpg".format(out_dir_videos + vid_id + '.mp4',
                                                                    out_dir_frames + '/')
        call(command_2, shell=True)
        '''
        out_dir_frames = curr_dir + '/../data/' + vid_id
        get_yahoo_score(out_dir_frames + '/', nsfw_net_yahoo)
        
    # adult_vid_scores.close()


def caffe_preprocess_and_compute(pimg, caffe_transformer=None, caffe_net=None,
                                 output_layers=None):
    """
    Run a Caffe network on an input image after preprocessing it to prepare
    it for Caffe.
    :param PIL.Image pimg:
        PIL image to be input into Caffe.
    :param caffe.Net caffe_net:
        A Caffe network with which to process pimg afrer preprocessing.
    :param list output_layers:
        A list of the names of the layers from caffe_net whose outputs are to
        to be returned.  If this is None, the default outputs for the network
        are returned.
    :return:
        Returns the requested outputs from the Caffe net.
    """
    if caffe_net is not None:

        # Grab the default output names if none were requested specifically.
        if output_layers is None:
            output_layers = caffe_net.outputs

        image = caffe.io.load_image(pimg)

        H, W, _ = image.shape
        _, _, h, w = caffe_net.blobs['data'].data.shape
        h_off = int(max((H - h) / 2, 0))
        w_off = int(max((W - w) / 2, 0))
        crop = image[h_off:h_off + h, w_off:w_off + w, :]
        transformed_image = caffe_transformer.preprocess('data', crop)
        transformed_image.shape = (1,) + transformed_image.shape

        input_name = caffe_net.inputs[0]
        all_outputs = caffe_net.forward_all(blobs=output_layers,
                                            **{input_name: transformed_image})

        outputs = all_outputs[output_layers[0]][0].astype(float)
        return outputs
    else:
        return []


def get_yahoo_score(dir_name, model_incep):
    # Load transformer
    # Note that the parameters are hard-coded for best results
    # fout = open("/home/kai/gm/video_frames/finetuned_result_5sep.txt", 'a+')
    caffe_transformer = caffe.io.Transformer({'data': nsfw_net_yahoo.blobs['data'].data.shape})
    caffe_transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost
    caffe_transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    caffe_transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    caffe_transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    _total = 0
    # _dir = dir_name.strip().split('/')[-2].strip()
    _dir = "result_testing"
    print("_dir = {}".format(_dir))
    res_dir = "/home/ubuntu/ML/user/Minhaj/inception_finetune/data/"
    if not os.path.exists(res_dir + str(_dir)):
        os.makedirs(res_dir + _dir)
        os.makedirs(res_dir + _dir + "/normal")
        os.makedirs(res_dir + _dir + '/adult_high')
        os.makedirs(res_dir + _dir + '/adult_low')

    len_files = len(os.listdir(dir_name))
    for i, _file in enumerate(os.listdir(dir_name)):
        try:
            image_data = dir_name + '/' + _file
            scores_yahoo = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer,
                                                        caffe_net=nsfw_net_yahoo,
                                                        output_layers=['prob'])
            # scores_p2k = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer,
            #                                             caffe_net=nsfw_net_2k,
            #                                             output_layers=['prob'])
            # scores_deep_miles = caffe_preprocess_and_compute(image_data, caffe_transformer=caffe_transformer,
            #                                             caffe_net=nsfw_net_deep_miles,
            #                                             output_layers=['prob'])
            # img = Image.open(image_data)
            # img = img.resize((299, 299), Image.ANTIALIAS)
            # img = np.reshape(img, [1, 299, 299, 3])
            # # img = img.convert('RGB')
            # x = np.asarray(img, dtype='float32')
            # # x/=255
            # incp_val = model_incep.predict(x)[0][0]
            
            value_yahoo = float(scores_yahoo[1])
            # value_p2k = float(scores_p2k[1])
            # value_deep_miles = float(scores_deep_miles[1])
            score_total = "yah_"+str("{:.2f}".format(value_yahoo))+str(_file)
            print ("{}/{}: {}".format(i, len_files, score_total))
            # max_age = scores.argmax(axis=0)
            # print(value_yahoo, _dir)

            # _count_ensemble = 0
            # if value_yahoo > 0.50:
            #     _count_ensemble += 1
            # _total += 1
            # if value_p2k > 0.50:
            #     _count_ensemble += 1
            # _total += 1
            # if value_deep_miles > 0.50:
            #     _count_ensemble += 1
            # _total += 1
            # if incp_val > 0.60:
            #     _count_ensemble += 1

            if incp_val >= 0.8 :
                shutil.copy(image_data,
                            res_dir + _dir +  "/adult_high/" + score_total)
            elif  incp_val >= 0.6 and value_yahoo < 0.8 :
                shutil.copy(image_data,
                            res_dir + _dir +  "/adult_low/" + score_total)
            else :
                shutil.copy(image_data,
                            res_dir + _dir +  "/normal/" + score_total)

            # adult_vid_scores.write(_file+", "+score_total+", "+value_yahoo+", "+value_p2k+", "+value_deep_miles+", "+incp_val, "\n")
            # avg_score = (value_yahoo + value_p2k + value_deep_miles + incp_val) / 4
            adult_vid_scores.write("{}, {}\n".format(_file, value_yahoo))
        except:
            classification_error.write(_file+"\n")

        '''    
        _total += 1
        if value_yahoo < 0.50:
            shutil.copy(image_data,
                        res_dir + _dir + "/" + score_total)
        elif value_yahoo > 0.50:
            shutil.copy(image_data,
                        res_dir + _dir + "/" + score_total)
        # adult_scores_arry.append(max_age)
        '''
    #_score_yahoo = float(float(_count_yahoo) / _total)
    # return score_total

adult_vid_scores = open("../files/yahoo_scores.txt", "w")
classification_error = open("../files/classification_error", "w")
all_images = ["testing_data"]
violence_adult_scores(all_images)
adult_vid_scores.close()
classification_error.close()
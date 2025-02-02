import tensorflow as tf
import numpy as np
import time as tm
import fixed_env as env
import load_trace as load_trace
import L2AC_ppo as ac
import os

# work parameter
DEBUG = False

# env parameter
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kpbs
latency_limit = 1.8
la_sum = 0

# env setting
random_seed = 10

# net parameter
S_DIM = 7
S_LEN = 8
A_DIM = 6
LA_DIM = 3
la_dict_list = [1.43, 1.62, 1.8]
LR_A = 0.0001
LR_LA = 0.0001
LR_C = 0.001

# QOE setting
reward_frame = 0
bitrate_util_total = 0
rebuff_p_total = 0
latency_p_total = 0
smooth_p_total = 0
skip_p_total = 0

reward_all = 0
bitrate_util_all =0
rebuff_p_all = 0
latency_p_all = 0
smooth_p_all = 0
skip_p_all = 0

SMOOTH_PENALTY = 0.02
REBUF_PENALTY = 1.85
SKIP_PENALTY = 0.5
LANTENCY_PENALTY = 0.005

# train path
#NN_MODEL = None
NN_MODEL = '/content/drive/MyDrive/L2AC/L2AC_results/nn_model_ep_49.ckpt' #  can load trained model
NETWORK_TRACE = ['fixed', 'high', 'low', 'medium', 'middle']
VIDEO_TRACE = 'AsianCup_China_Uzbekistan'
VIDEO_TRACE_list = ['AsianCup_China_Uzbekistan', 'Fengtimo_2018_11_3', 'game', 'room', 'sports']
#network_trace_dir = './dataset/network_trace/' + NETWORK_TRACE + '/'
#video_trace_prefix = './dataset/video_trace/' + VIDEO_TRACE + '/frame_trace_'
#LOG_FILE_PATH = '/content/drive/MyDrive/L2AC/log/train'
SUMMARY_DIR = '/content/drive/MyDrive/L2AC/L2AC_results'  # trained model path
LOG_FILE_PATH = './log/test'
#SUMMARY_DIR = './L2AC_results'  # trained model path

# load the network trace
#all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)

# defalut setting
epoch_reward = 0
epoch_bitrate_util = 0
epoch_rebuff_p = 0
epoch_latency_p = 0
epoch_smooth_p = 0
epoch_skip_p = 0
last_bit_rate = 0
bit_rate = 0
target_buffer = 0
state = np.zeros((S_DIM, S_LEN))
thr_record = np.zeros(8)

# plot info
idx = 0
id_list = []
bit_rate_record = []
buffer_record = []
throughput_record = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    actor = ac.Actor(sess, n_features=[S_DIM, S_LEN], n_actions=A_DIM, lr=LR_A)
    critic = ac.Critic(sess, n_features=[S_DIM, S_LEN], lr=LR_C)
    L_actor = ac.LActor(sess, n_features=[S_DIM, S_LEN], n_actions=LA_DIM, lr=LR_LA)
    sess.run(tf.global_variables_initializer())

    # reader = pywrap_tensorflow.NewCheckpointReader("./submit/results/nn_model_ep_ac_1.ckpt")
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print(key)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        exclude=['train/LActor', 'LActor', 'train_2/beta1_power', 'train_2/beta2_power'])
    saver1 = tf.train.Saver(max_to_keep=200)  # save neural net parameters
    saver2 = tf.train.Saver(max_to_keep=200)  # save neural net parameters
    nn_model = NN_MODEL

    meta_file = NN_MODEL + '.meta'
    saver = tf.train.import_meta_graph(meta_file)
    if nn_model is not None:  # nn_model is the path to file
        saver.restore(sess, nn_model)
        print("Model restored.")
    
    for network in NETWORK_TRACE:
        print('network: ', network)
        for video in VIDEO_TRACE_list:
            print('video: ', video)
            chunk_reward = 0
            video_count = 0
            is_first = True

            #VIDEO_TRACE = VIDEO_TRACE_list[0]
            video_trace_prefix = './dataset/video_trace/' + video + '/frame_trace_'
            network_trace_dir = './dataset/network_trace/' + network + '/'
            all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(network_trace_dir)
            net_env = env.Environment(all_cooked_time=all_cooked_time,
                                      all_cooked_bw=all_cooked_bw,
                                      random_seed=random_seed,
                                      logfile_path=LOG_FILE_PATH,
                                      VIDEO_SIZE_FILE=video_trace_prefix,
                                      Debug=DEBUG)

            log_folder = LOG_FILE_PATH + '/' + network + '/' + video
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            log_path = log_folder + '/' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'a')

            pre_ac = 0
            while True:
                timestamp_start = tm.time()
                reward_frame = 0
                bitrate_util_total = 0
                rebuff_p_total = 0
                latency_p_total = 0
                smooth_p_total = 0
                skip_p_total = 0
                
                

                time, time_interval, send_data_size, frame_time_len, \
                rebuf, buffer_size, play_time_len, end_delay, \
                cdn_newest_id, download_id, cdn_has_frame, skip_frame_time_len, decision_flag, \
                buffer_flag, cdn_flag, skip_flag, end_of_video = net_env.get_video_frame(bit_rate, target_buffer, latency_limit)

                pre_bit_rate = bit_rate
                pre_latency_limit = latency_limit

                # QOE setting
                if end_delay <= 1.0:
                    LANTENCY_PENALTY = 0.005
                else:
                    LANTENCY_PENALTY = 0.01

                if not cdn_flag:
                    bitrate_util = frame_time_len * float(BIT_RATE[bit_rate]) / 1000
                    rebuff_p = REBUF_PENALTY * rebuf
                    latency_p = LANTENCY_PENALTY * end_delay
                    skip_p = SKIP_PENALTY * skip_frame_time_len
                    
                    reward_frame = bitrate_util - rebuff_p - latency_p - skip_p
                    
                    bitrate_util_total += bitrate_util
                    rebuff_p_total += rebuff_p
                    latency_p_total += latency_p
                    skip_p_total += skip_p
                    
                    
                else:
                    rebuff_p = REBUF_PENALTY * rebuf
                    reward_frame = -(REBUF_PENALTY * rebuf)
                    
                    rebuff_p_total += rebuff_p
                    

                chunk_reward += reward_frame

                if decision_flag or end_of_video:
                    
                    smooth_p = SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
                    reward_frame += -1 * smooth_p
                    chunk_reward += -1 * smooth_p
                    
                    smooth_p_total += smooth_p
                    # last_bit_rate

                    reward = chunk_reward
                    chunk_reward = 0

                    # ----------------- the Algorithm ---------------------

                    if not cdn_flag and time_interval is not 0:
                        thr = send_data_size / time_interval / 1000000
                    else:
                        thr = thr

                    thr_record = np.roll(thr_record, -1, axis=0)
                    thr_record[-1] = thr
                    thr_mean = np.mean(thr_record[-4:])
                    thr_variance = np.var(thr_record[-4:])

                    state = np.roll(state, -1, axis=1)
                    # State
                    state[0, -1] = buffer_size / 10.0
                    state[1, -1] = thr / 10.0
                    state[2, -1] = len(cdn_has_frame[0]) / 40.0
                    state[3, -1:] = bit_rate / 10.0
                    state[4, -1] = end_delay / 10.0
                    state[5, -1] = skip_frame_time_len / 10.0
                    state[6, -1] = rebuf / 10.0
                    # other tried features:
                    # state[7, -1] = time_interval / 10.0
                    # state[8, -1] = thr_variance / 10.0
                    # state[9, -1] = skip_frame_time_len

                    action, a_probs = actor.choose_action(state)
                    action_vec = np.zeros(A_DIM)
                    action_vec[action] = 1
                    action_vec = np.expand_dims(action_vec, 0)

                    laction, la_probs = L_actor.choose_action(state)  # latency network action
                    laction_vec = np.zeros(LA_DIM)
                    laction_vec[laction] = 1
                    laction_vec = np.expand_dims(laction_vec, 0)

                    latency_limit = la_dict_list[laction]
                    la_sum += 1
                    if action == 0:
                        bit_rate = 0
                        target_buffer = 1
                    if action == 1:
                        bit_rate = 1
                        target_buffer = 1
                    if action == 2:
                        bit_rate = 0
                        target_buffer = 0
                    if action == 3:
                        bit_rate = 1
                        target_buffer = 0
                    if action == 4:
                        bit_rate = 2
                        target_buffer = 0
                    if action == 5:
                        bit_rate = 3
                        target_buffer = 0


                    if is_first:
                        is_first = False

                    pre_state = state
                    pre_ac = action_vec
                    pre_la_ac = laction_vec
                    pre_ac_probs = a_probs
                    pre_lac_probs = la_probs
                    last_bit_rate = bit_rate

                reward_all += reward_frame
                bitrate_util_all += bitrate_util_total
                rebuff_p_all += rebuff_p_total
                latency_p_all += latency_p_total
                smooth_p_all += smooth_p_total
                skip_p_all += skip_p_total

                log_file.write(str(time) + '\t' +
                                   str(time_interval) + '\t' +
                                   str(BIT_RATE[pre_bit_rate]) + '\t' +
                                   str(pre_latency_limit) + '\t' +
                                   str(target_buffer) + '\t' +
                                   str(frame_time_len) + '\t' +
                                   str(buffer_size) + '\t' +
                                   str(rebuf) + '\t' +
                                   str(send_data_size) + '\t' +
                                   str(end_delay) + '\t' +
                                   str(skip_frame_time_len) + '\t' +
                                   str(reward_all) + '\n')
                log_file.flush()
                if end_of_video:
                    log_file.write('\n')
                    log_file.close()
                    print("network traceID: %d, network_reward: %f, avg_running_time: %f" %
                          (video_count,
                           reward_all,
                           tm.time() - timestamp_start))
                    epoch_reward += reward_all
                    epoch_bitrate_util += bitrate_util_all
                    epoch_rebuff_p += rebuff_p_all
                    epoch_latency_p += latency_p_all
                    epoch_smooth_p += smooth_p_all
                    epoch_skip_p += skip_p_all
                    reward_all = 0
                    bitrate_util_all =0
                    rebuff_p_all = 0
                    latency_p_all = 0
                    smooth_p_all = 0
                    skip_p_all = 0
                    video_count += 1
                    if video_count >= len(all_file_names):
                        print("epoch total reward: %f" % (epoch_reward / video_count))
                        print("bitrate util: %f" % (epoch_bitrate_util / video_count))
                        print("rebuff penalty: %f" % (epoch_rebuff_p / video_count))
                        print("latency penalty: %f" % (epoch_latency_p / video_count)) 
                        print("smooth penalty: %f" % (epoch_smooth_p / video_count))
                        print("skip penalty: %f" % (epoch_skip_p / video_count))
                        epoch_reward = 0
                        epoch_bitrate_util = 0
                        epoch_rebuff_p = 0
                        epoch_latency_p = 0
                        epoch_smooth_p = 0
                        epoch_skip_p = 0
                        break

                log_path = log_folder + '/' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'a')

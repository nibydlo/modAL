import pickle
import matplotlib.pyplot as plt

from visualization.plots import plot_conf_int

passive_e1_accs = []
# passive_e5_accs = []
# passive_e10_accs = []
#
# passive_ae_e1_accs = []
# passive_ae_es01_t1_accs = []
# passive_ae_es01_t2_accs = []
#
# ll_1_accs = []
# ll_2_accs = []
# ll_3_accs = []
#
# ll2_margin_1_accs = []
# ll2_margin_2_accs = []
# ll2_margin_3_accs = []
#
# ll2_margin_only_accs = []
#
# ll2_1_margin_hidden1_accs = []
# ll2_1_margin_hidden2_accs = []
# ll2_1_margin_hidden3_accs = []
#
# ll2_2_margin_accs = []
# ll2_2_exp_margin_accs = []
#
# ll3_margin_bald_1_accs = []
#
# ll3_1_margin_bald_hidden1_accs = []
# ll3_1_margin_bald_hidden2_accs = []
#
# ll4_margin_2_accs = []

ll_ideal_accs = []

margin_accs = []

min_n_queries = None

for i in range(2, 3):

    # state = pickle.load(open('statistic/topics/torch_bn/passive_e1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_e1_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch_bn/passive_e5_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_e5_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/passive_e10_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_e10_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_ae/passive_ae_e1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_ae_e1_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_ae/passive_ae_es01_tol1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_ae_es01_t1_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_ae/passive_ae_es01_tol2_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # passive_ae_es01_t2_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/learning_loss_1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll_1_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/learning_loss_2_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll_2_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/learning_loss_3_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll_3_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.0_margin_n_hidden1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll2_margin_1_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.0_margin_n_hidden2_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll2_margin_2_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.0_margin_n_hidden3_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll2_margin_3_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.0_margin_only_n_hidden1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll2_margin_only_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/margin_inter_i2000_b20_q100_' + str(i) + '.pkl', 'rb'))
    # margin_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch_bn/ll3_margin_bald_n_hidden1_i2000_b20_q200_test' + str(i) + '.pkl', 'rb'))
    # ll3_margin_bald_1_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.1_margin_n_hidden1_i2000_b20_q200_test' + str(i) + '.pkl', 'rb'))
    # ll2_1_margin_hidden1_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.1_margin_n_hidden2_i2000_b20_q200_test' + str(i) + '.pkl', 'rb'))
    # ll2_1_margin_hidden2_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.1_margin_n_hidden3_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll2_1_margin_hidden3_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch_bn/ll2.2_margin_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll2_2_margin_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll2.2_exp_margin_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll2_2_exp_margin_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch_bn/ll3.1_margin_bald_n_hidden1_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll3_1_margin_bald_hidden1_accs.append(state['performance_history'])
    #
    # state = pickle.load(open('statistic/topics/torch_bn/ll3.1_margin_bald_n_hidden2_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll3_1_margin_bald_hidden2_accs.append(state['performance_history'])

    # state = pickle.load(open('statistic/topics/torch_bn/ll4.0_margin_n_hidden2_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    # ll4_margin_2_accs.append(state['performance_history'])

    state = pickle.load(open('statistic/topics/torch_bn/ll_ideal_i2000_b20_q200_' + str(i) + '.pkl', 'rb'))
    ll_ideal_accs.append(state['performance_history'])

n_queries = len(state['performance_history']) - 1
plot_conf_int(passive_e1_accs, 2000, 20, n_queries, 'passive_e1', color='C0')
plot_conf_int(margin_accs, 2000, 20, n_queries, 'margin', color='C2')

# plot_conf_int(passive_e5_accs, 2000, 20, n_queries, 'passive_e5', color='C1')
# plot_conf_int(passive_e10_accs, 2000, 20, n_queries, 'passive_e10', color='C2')
# plot_conf_int(passive_ae_e1_accs, 2000, 20, n_queries, 'passive_ae_e1', color='C3')
# plot_conf_int(passive_ae_es01_t1_accs, 2000, 20, n_queries, 'passive_ae_es01_t1', color='C4')
# plot_conf_int(passive_ae_es01_t2_accs, 2000, 20, n_queries, 'passive_ae_es01_t2', color='C5')

# plot_conf_int(ll_1_accs, 2000, 20, n_queries, 'learning loss 1', color='C6')
# plot_conf_int(ll_2_accs, 2000, 20, n_queries, 'learning loss 2', color='C7')
# plot_conf_int(ll_3_accs, 2000, 20, n_queries, 'learning loss 3', color='C8')
#
# plot_conf_int(ll2_margin_1_accs, 2000, 20, n_queries, 'll2 margin 1', color='C9')
# plot_conf_int(ll2_margin_2_accs, 2000, 20, n_queries, 'll2 margin 2', color='C10')
# plot_conf_int(ll2_margin_3_accs, 2000, 20, n_queries, 'll2 margin 3', color='C11')

# plot_conf_int(ll2_margin_only_accs, 2000, 20, n_queries, 'll2 margin only', color='C13')
# plot_conf_int(ll3_margin_bald_1_accs, 2000, 20, n_queries, 'll3 margin bald 1', color='C14')

# plot_conf_int(ll2_1_margin_hidden1_accs, 2000, 20, n_queries, 'll2.1 margin 1', color='C6')
# plot_conf_int(ll2_1_margin_hidden2_accs, 2000, 20, n_queries, 'll2.1 margin 2', color='C7')
# plot_conf_int(ll2_1_margin_hidden3_accs, 2000, 20, n_queries, 'll2.1 margin 3', color='C8')
# plot_conf_int(ll2_2_margin_accs, 2000, 20, n_queries, 'll2.2', color='C8')
# plot_conf_int(ll2_2_exp_margin_accs, 2000, 20, n_queries, 'll2.2', color='C9')
# plot_conf_int(ll3_1_margin_bald_hidden1_accs, 2000, 20, 100, 'll3.1 margin bald 1', color='C9')
# plot_conf_int(ll3_1_margin_bald_hidden2_accs, 2000, 20, 200, 'll3.1 margin bald 2', color='C11')
# plot_conf_int(ll4_margin_2_accs, 2000, 20, 200, 'll4.0 margin 2', color='C11')
plot_conf_int(ll_ideal_accs, 2000, 20, n_queries, 'll ideal', color='C11')


plt.xlabel('labeled size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()

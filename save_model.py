from SI_Toolkit.TF.Parameters import args
from SI_Toolkit.TF.TF_Functions.Initialization import set_seed, create_full_name, create_log_file, \
    get_net, get_norm_info_for_net

print('')
a = args()  # 'a' like arguments
print(a.__dict__)
print('')
set_seed(a)
net, net_info = get_net(a)
print(net.summary())
save_loc = a.path_to_models + a.net_name + '/saved_model.h5'
print('Saving Model: ', save_loc)
net.save(save_loc)

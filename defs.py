'''
FILE NECESARY TO HAVE ALL SETUPS(I.E. ALL DEFINITIONS) IN ONE EASY TO FIND PLACE, IF FOR WHATEVER REASON THIS MODIFICATION DOES NOT WORK THEN
THE NEEDED MODIFICATIONS ARE THE FOLLOWING:
    UNCOMMENT THE FOLLOWING LINES IN RESPECTIVE FILES:
                            lines 15-19 train.py
                            lines 5-8 avg_loss_plot.py
                            lines 13-19 test.py
                            lines 5-9 and lines 167-170 graph_conv.py
'''
import os

#PARAMETERS FOR TRAINING
num_epochs =  1
restart_train_from = 149
minibatch_size = 128
dropout_keep = 0.5
filters_number = 256
#PARAMETERS FOR LOSS
pn_ratio = 0.1
learning_rate = 0.05

#PARAMETERS FOR TESTING
model_num = 0
model_list = [40,45,50,55,60,65,70,80,100,110,130,140]

'''THESE LINES NEED TO BE UNCOMMENTED FOR GENERAL USE'''
train_data_file = os.path.join('.\data_SimpleVersion', 'train.cpkl')
test_data_file = os.path.join('.\data_SimpleVersion', 'test.cpkl')
no_edge = "./avg_loss_train.txt"
edge = "./avg_loss_train_EDGE.txt"
no_edge_test = "./CONTROL_Testing_loss_noedge.txt"
edge_test = "./Testing_loss.txt"

'''THESE LINES NEED TO BE UNCOMMENTED IF GRETA IS USING THIS()'''
'''
train_data_file = os.path.join(
    'C:\\Users\\Cobal\\Desktop\\ComplexNetworksEXAM\\Graph_convolution_with_proteins-master\\data_SimpleVersion',
     'train.cpkl')
test_data_file = os.path.join(
    'C:\\Users\\Cobal\\Desktop\\ComplexNetworksEXAM\\Graph_convolution_with_proteins-master\\data_SimpleVersion',
    'test.cpkl')
no_edge = os.path.join(
    'C:\\Users\\Cobal\\Desktop\\ComplexNetworksEXAM\\Graph_convolution_with_proteins-master\\data_SimpleVersion',
     'avg_loss_train.txt')
edge = os.path.join(
    'C:\\Users\\Cobal\\Desktop\\ComplexNetworksEXAM\\Graph_convolution_with_proteins-master\\data_SimpleVersion',
    'avg_loss_train_EDGE.txt')
'''

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
num_epochs =  150
minibatch_size = 128
dropout_keep = 0.5
#PARAMETERS FOR LOSS
pn_ratio = 0.1
learning_rate = 0.05

#PARAMETERS FOR TESTING
model_list = [0,29,59,89,119,149]

'''THESE LINES NEED TO BE UNCOMMENTED FOR GENERAL USE'''
train_data_file = os.path.join('.\data_SimpleVersion', 'train.cpkl')
test_data_file = os.path.join('.\data_SimpleVersion', 'test.cpkl')
no_edge = "./avg_loss_train.txt"
edge = "./avg_loss_train_EDGE.txt"

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

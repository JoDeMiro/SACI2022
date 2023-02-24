import argparse

def main():
    parser = argparse.ArgumentParser(description='Evolution train musz-musz description')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA and run on CPU.')
    # Dataset
    parser.add_argument('--dataset', type=str, choices = ['EURUSD', '56', 'SYNTHETIC', 'SINUS'], default='56', help='Choice of the dataset. Synthetic datasets must have been generated previously with synth_dataset_gen.py. Default: 56.')
    # Training
    parser.add_argument('--train-mode', choices = ['BP','FA','DFA','DRTP','sDFA','shallow'], default='DRTP', help='Choice of the training algorithm - backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), direct random target propagation (DRTP), error-sign-based DFA (sDFA), shallow learning with all layers freezed but the last one that is BP-trained (shallow). Default: DRTP.')
    parser.add_argument('--optimizer', choices = ['SGD', 'NAG', 'Adam', 'RMSprop'], default='NAG', help='Choice of the optimizer - stochastic gradient descent with 0.9 momentum (SGD), SGD with 0.9 momentum and Nesterov-accelerated gradients (NAG), Adam (Adam), and RMSprop (RMSprop). Default: NAG.')
    parser.add_argument('--loss', choices = ['MSE', 'BCE', 'CE'], default='BCE', help='Choice of loss function - mean squared error (MSE), binary cross entropy (BCE), cross entropy (CE, which already contains a logsoftmax activation function). Default: BCE.')
    parser.add_argument('--freeze-conv-layers', action='store_true', default=False, help='Disable training of convolutional layers and keeps the weights at their initialized values.')
    parser.add_argument('--fc-zero-init', action='store_true', default=False, help='Initializes fully-connected weights to zero instead of the default He uniform initialization.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout probability (applied only to fully-connected layers). Default: 0.')
    parser.add_argument('--trials', type=int, default=1, help='Number of training trials Default: 1.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs Default: 100.')
    parser.add_argument('--batch-size', type=int, default=100, help='Input batch size for training. Default: 100.')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Input batch size for testing Default: 1000.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4.')
    # Network
    parser.add_argument('--topology', type=str, default='CONV_32_5_1_2_FC_1000_FC_10', help='Choice of network topology. Format for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}. Format for fully-connected layers: FC_{output units}.')
    parser.add_argument('--conv-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the convolutional layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--hidden-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the fully-connected hidden layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--output-act', type=str, choices = {'sigmoid', 'tanh', 'none'}, default='sigmoid', help='Type of activation for the network output layer - Sigmoid (sigmoid), Tanh (tanh), none (none). Default: sigmoid.')

    args = parser.parse_args()

    print(args)

    # (device, train_loader, traintest_loader, test_loader) = setup.setup(args)

    # train.train(args, device, train_loader, traintest_loader, test_loader)

if __name__ == '__main__':
    main()

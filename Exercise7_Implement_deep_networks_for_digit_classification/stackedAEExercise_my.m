 clear all; close all;


% inputSize = 28 * 28;
% numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       


trainData = loadMNISTImages('../mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('../mnist/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1


testData = loadMNISTImages('../mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('../mnist/t10k-labels-idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

% [acc] = stackedAutoencoders(trainData, testData, trainLabels, testLabels, hiddenSizeL1,hiddenSizeL2,sparsityParam,lambda, beta);

hiddenSize = 200;
[pred] = myNeuralNetworkClassification(trainData,testData,trainLabels, hiddenSize, 0.0001,200);

mean(pred == testLabels)

% [theta, acc_train,pred_test,acc_test,cost] = my_softmax_regression2(trainData',trainLabels,testData',testLabels,0.0001,200);
% acc_test
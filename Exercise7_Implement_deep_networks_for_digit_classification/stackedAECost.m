function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.


L = length(stack); % number of hidder layers
z = cell(L,1);

for d = 1:L
%     a{d} = zeros(size(stack{d}.w,1),n);
    if d == 1
        z{d} = bsxfun(@plus, stack{1}.w*data, stack{1}.b);
    else
        z{d} = bsxfun(@plus, stack{d}.w*sigmoid(z{d-1}), stack{d}.b);
    end
end

aOut = softmaxTheta * sigmoid(z{L});
aOut = exp(aOut);
aOut = bsxfun(@rdivide, aOut, sum(aOut));

% z2 = bsxfun(@plus, stack{1}.w*data, stack{1}.b);
% a2 = sigmoid(z2);
% z3 = bsxfun(@plus, stack{2}.w*a2, stack{2}.b);
% a3 = sigmoid(z3);


delta = cell(L+2,1); % delta{1} is not used
for d = L+2:-1:1
    if d == L+2
        delta{d} = -(groundTruth - aOut);
    elseif d == L+1
        delta{d} = (softmaxTheta' * delta{d+1}) .* sigmoidGrad(z{d-1});
    else
        delta{d} = (stack{2}.w' * delta{d+1}) .* sigmoidGrad(z{d-1});
    end
end


% z4 = softmaxTheta * a3;
% a4 = exp(z4);
% a4 = bsxfun(@rdivide, a4, sum(a4));

% delta4 = -(groundTruth - a4);
% delta3 = (softmaxTheta' * delta4) .* sigmoidGrad(z3);
% delta2 = (stack{2}.w' * delta3) .* sigmoidGrad(z2);

% softmaxThetaGrad = -(1/ M) * (groundTruth - a4) * a3'  + lambda * softmaxTheta;
softmaxThetaGrad = -(1/M)*delta{L+2} * sigmoid(z{L})'  + lambda * softmaxTheta;

for d = L:-1:1
    if d == 1
       stackgrad{1}.w = (1/M) * delta{2} * data' + lambda * stack{1}.w;
       stackgrad{1}.b = (1/M) * sum(delta{2}, 2); 
    else
        stackgrad{d}.w = (1/M) * delta{d+1} * a2' + lambda * stack{d}.w;
        stackgrad{d}.b = (1/M) * sum(delta{d+1}, 2);
    end
end

% stackgrad{2}.w = (1. / M) * delta3 * a2' + lambda * stack{2}.w;
% stackgrad{2}.b = (1. / M) * sum(delta3, 2);
% stackgrad{1}.w = (1. / M) * delta2 * data' + lambda * stack{1}.w;
% stackgrad{1}.b = (1. / M) * sum(delta2, 2);

% cost = (1. / M) * sum((1. / 2) * sum((a4 - groundTruth).^2));
cost = -(1/M)*sum(sum(groundTruth .* log(aOut))) + (lambda/2)*sum(sum(softmaxTheta.^2)) ;

for d = 1:L
    cost = cost + (lambda/2)*(sum(sum(stack{d}.w .^2)));
end

% cost = -(1. / M) * sum(sum(groundTruth .* log(a4))) + (lambda / 2.) * ...
%     sum(sum(softmaxTheta.^2)) + (lambda / 2.) * (sum(sum(stack{1}.w .^2)) ...
%     + sum(sum(stack{2}.w .^2)));



% deltaW1 = delta2 * a1';
% deltab1 = sum(delta2, 2);
% deltaW2 = delta3 * a2';
% deltab2 = sum(delta3, 2);
% 
% 
% W1grad = (1. / m) * deltaW1 + lambda * W1;
% b1grad = (1. / m) * deltab1;
% W2grad = (1. / m) * deltaW2 + lambda * W2;
% b2grad = (1. / m) * deltab2;





function grad = softmaxGrad(x)
    e_x = exp(-x);
    grad = e_x ./ (1 + (1-e_x).*e_x ).^2;
end

function grad = sigmoidGrad(x)
    e_x = exp(-x);
    grad = e_x ./ ((1 + e_x).^2); 
end


% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

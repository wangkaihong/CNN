function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,filterDim,numFilters,poolDim,hiddenDim,pred)

if ~exist('pred','var')
    pred = false;
end;

weightDecay = 0.0001;

imageDim = size(images,1); 
numImages = size(images,3); 

[Wc , Wh , Wd, bc, bh , bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,poolDim,numClasses,hiddenDim);

Wc_grad = zeros(size(Wc));
Wh_grad = zeros(size(Wh));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bh_grad = zeros(size(bh));
bd_grad = zeros(size(bd));

convDim = imageDim-filterDim+1; 
outputDim = (convDim)/poolDim;

activations = zeros(convDim,convDim,numFilters,numImages);

activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

activations = cnnConvolve(filterDim, numFilters, images, Wc, bc);
activationsPooled = cnnPool(poolDim, activations);
 

activationsPooled = reshape(activationsPooled,[],numImages);

probs = zeros(numClasses,numImages);
hidden = Wh * activationsPooled;
hidden = sigmoid(bsxfun(@plus,hidden,bh));

z = Wd*hidden;
z = bsxfun(@plus,z,bd);

z = bsxfun(@minus,z,max(z,[],1));
z = exp(z);
probs = bsxfun(@rdivide,z,sum(z,1));
preds = probs;

cost = 0; 

logProbs = log(probs);
labelIndex=sub2ind(size(logProbs), labels', 1:size(logProbs,2));

values = logProbs(labelIndex);
cost = -sum(values);
weightDecayCost = (weightDecay/2) * (sum(Wd(:) .^ 2) + sum(Wh(:) .^ 2) + sum(Wc(:) .^ 2));
cost = cost / numImages+weightDecayCost; 
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;



targetMatrix = zeros(size(probs));  
targetMatrix(labelIndex) = 1;  
softmaxError = probs-targetMatrix;

hiddenError = (Wd' * softmaxError) .* hidden .* (1 - hidden);

poolError = Wh'*hiddenError;
poolError = reshape(poolError, outputDim, outputDim, numFilters, numImages);

unpoolError = zeros(convDim, convDim, numFilters, numImages);
unpoolingFilter = ones(poolDim);
poolArea = poolDim*poolDim;

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        e = poolError(:, :, filterNum, imageNum);
        unpoolError(:, :, filterNum, imageNum) = kron(e, unpoolingFilter)./poolArea;
    end
end

convError = unpoolError .* activations .* (1 - activations); 


Wd_grad = (1/numImages).*softmaxError * hidden'+weightDecay * Wd;
bd_grad = (1/numImages).*sum(softmaxError, 2);

Wh_grad = (1/numImages).*hiddenError * activationsPooled'+weightDecay * Wh;
bh_grad = (1/numImages).*sum(hiddenError, 2);

bc_grad = zeros(size(bc));
Wc_grad = zeros(size(Wc));

for filterNum = 1 : numFilters
    e = convError(:, :, filterNum, :);
    bc_grad(filterNum) = (1/numImages).*sum(e(:));
end

for filterNum = 1 : numFilters
    for imageNum = 1 : numImages
        e = convError(:, :, filterNum, imageNum);
        convError(:, :, filterNum, imageNum) = rot90(e, 2);
    end
end

for filterNum = 1 : numFilters
    Wc_gradFilter = zeros(size(Wc_grad, 1), size(Wc_grad, 2));
    for imageNum = 1 : numImages     
        Wc_gradFilter = Wc_gradFilter + conv2(images(:, :, imageNum), convError(:, :, filterNum, imageNum), 'valid');
    end
    Wc_grad(:, :, filterNum) = (1/numImages).*Wc_gradFilter;
end
Wc_grad = Wc_grad + weightDecay * Wc;

grad = [Wc_grad(:) ; Wh_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bh_grad(:) ; bd_grad(:)];

end

tic

imageDim = 28;
numClasses = 10;
filterDim = 9;
numFilters = 15;
poolDim = 2; 
hiddenDim = 5;

addpath '';
images = loadMNISTImages('');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('');
labels(labels==0) = 10;

theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses,hiddenDim);



options.epochs = 4;
options.minibatch = 64;
options.alpha = 1e-1;
options.momentum = .95;

opttheta = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,filterDim,numFilters,poolDim,hiddenDim),theta,images,labels,options);


testImages = loadMNISTImages('');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('');
testLabels(testLabels==0) = 10; 

[cost,grad,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim,hiddenDim,true);
            
fprintf(' Cost is %f\n',cost);
acc = sum(preds==testLabels)/length(preds);

fprintf('Accuracy is %f\n',acc);
toc

function [Wc, Wh , Wd, bc , bh , bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,poolDim,numClasses,hiddenDim)
outDim = (imageDim - filterDim + 1)/poolDim;
totalOutPut = outDim^2*numFilters;

indS = 1;
indE = filterDim^2*numFilters;
Wc = reshape(theta(indS:indE),filterDim,filterDim,numFilters);

indS = indE+1;
indE = indE+totalOutPut*hiddenDim;
Wh = reshape(theta(indS:indE),hiddenDim,totalOutPut);

indS = indE+1;
indE = indE+hiddenDim*numClasses;
Wd = reshape(theta(indS:indE),numClasses,hiddenDim);

indS = indE+1;
indE = indE+numFilters;
bc = theta(indS:indE);

indS = indE+1;
indE = indE+hiddenDim;
bh = theta(indS:indE);

bd = theta(indE+1:end);


end
function theta = cnnInitParams(imageDim,filterDim,numFilters,poolDim,numClasses,hiddenDim)
assert(filterDim < imageDim,'filterDim must be less that imageDim');

Wc = 1e-1*randn(filterDim,filterDim,numFilters);

outDim = imageDim - filterDim + 1; 

assert(mod(outDim,poolDim)==0,...
       'poolDim must divide imageDim - filterDim + 1');

outDim = outDim/poolDim;
totalOutPut = outDim^2*numFilters;

r  = sqrt(6) / sqrt(numClasses+totalOutPut+1);
Wh = rand(hiddenDim, totalOutPut) * 2 * r - r;
Wd = rand(numClasses, hiddenDim) * 2 * r - r;

bc = zeros(numFilters, 1);
bh = zeros(hiddenDim,1);
bd = zeros(numClasses, 1);

theta = [Wc(:) ; Wh(:) ; Wd(:) ; bc(:) ; bh(:) ; bd(:)];

end
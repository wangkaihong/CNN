function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)

numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);


for imageNum = 1:numImages
  for filterNum = 1:numFilters

    convolvedImage = zeros(convDim, convDim);
    
    filter = W(:,:,filterNum);
    
    filter = rot90(squeeze(filter),2);
          
    im = squeeze(images(:, :, imageNum));

    
    convolvedImage = conv2(im,filter,'valid');
    convolvedImage = convolvedImage + (b(filterNum) * ones(convDim, convDim));
    convolvedImage = sigmoid(convolvedImage);
    
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end


end


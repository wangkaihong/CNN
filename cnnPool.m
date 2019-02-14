function pooledFeatures = cnnPool(poolDim, convolvedFeatures)

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim,convolvedDim / poolDim, numFilters, numImages);

    for numImage = 1:numImages
        for numFeature = 1:numFilters
            for poolRow = 1:convolvedDim / poolDim
                offsetRow = 1+(poolRow-1)*poolDim;
                for poolCol = 1:convolvedDim / poolDim
                    offsetCol = 1+(poolCol-1)*poolDim;
                    patch = convolvedFeatures(offsetRow:offsetRow+poolDim-1,offsetCol:offsetCol+poolDim-1,numFeature,numImage); 
                    pooledFeatures(poolRow,poolCol,numFeature,numImage) = mean(patch(:));
                end
            end            
        end
    end
    
end
function convolvedFeatures = cnnConvolve_my(patchDim, numFeatures, images, W, b, ZCAWhite, meanPatch)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  patchDim - patch (feature) dimension
%  numFeatures - number of features
%  images - large images to convolve with, matrix in the form
%           images(r, c, channel, image number)
%  W, b - W, b for features from the sparse autoencoder
%  ZCAWhite, meanPatch - ZCAWhitening and meanPatch matrices used for
%                        preprocessing
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)


%% this implementation is correct, but slow. Hence, use conv2 to speed up, which is 6x faster

numImages = size(images, 4);
imageDim = size(images, 1);
imageChannels = size(images, 3);


convolvedFeatures = zeros(numFeatures, numImages, imageDim - patchDim + 1, imageDim - patchDim + 1);

patchRow = imageDim - patchDim + 1;
patchColumn = imageDim - patchDim + 1;
for imageNum = 1:numImages
    for patchr = 1:patchRow
        for patchc = 1:patchColumn
            patch = images(patchr:patchr+patchDim-1, patchc:patchc+patchDim-1,:,imageNum);
            convolvedFeatures(:,imageNum,patchr, patchc) = sigmoid( W*ZCAWhite*(patch(:) - meanPatch) + b );
        end
    end
end


end


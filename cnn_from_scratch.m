numClasses = 2;

layers = [
    imageInputLayer([128 128 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

inputSize = layers(1).InputSize;
analyzeNetwork(layers);

trainImageDS = imageDatastore('ConcreteFaultSubset/train', 'IncludeSubfolders',true, 'LabelSource','foldernames');
valImageDS = imageDatastore('ConcreteFaultSubset/val', 'IncludeSubfolders',true, 'LabelSource','foldernames');

miniBatchSize = 10;
augTrain = augmentedImageDatastore(inputSize(1:2), trainImageDS);
augVal = augmentedImageDatastore(inputSize(1:2), valImageDS);

options = trainingOptions('adam', 'MiniBatchSize',miniBatchSize, 'MaxEpochs',6, 'InitialLearnRate',3e-4, 'Shuffle','every-epoch', 'Plots','training-progress', 'ValidationData',augVal, 'ValidationFrequency',3);
net = trainNetwork(augTrain,layers,options);

[YPred, probs] = classify(net, augVal);
accuracy = mean(YPred == valImageDS.Labels)
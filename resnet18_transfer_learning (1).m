net = resnet18();
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);

net.Layers

numClasses = 2;

oldFcLayer = lgraph.Layers(69);
oldClassLayer = lgraph.Layers(71);

newFcLayer = fullyConnectedLayer(numClasses,'Name','new_fc', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10);
newClassLayer = classificationLayer('Name','new_classoutput');

lgraph = replaceLayer(lgraph, oldFcLayer.Name, newFcLayer);
lgraph = replaceLayer(lgraph, oldClassLayer.Name, newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:68) = freezeWeights(layers(1:68));
lgraph = createLgraphUsingConnections(layers,connections);
lgraph.Layers

trainImageDS = imageDatastore('ConcreteFaultSubset/train', 'IncludeSubfolders',true, 'LabelSource','foldernames');
valImageDS = imageDatastore('ConcreteFaultSubset/val', 'IncludeSubfolders',true, 'LabelSource','foldernames');

miniBatchSize = 10;
augTrain = augmentedImageDatastore(inputSize(1:2), trainImageDS);
augVal = augmentedImageDatastore(inputSize(1:2), valImageDS);

options = trainingOptions('adam', 'MiniBatchSize',miniBatchSize, 'MaxEpochs',6, 'InitialLearnRate',3e-4, 'Shuffle','every-epoch', 'Plots','training-progress', 'ValidationData',augVal, 'ValidationFrequency',3);
net = trainNetwork(augTrain,lgraph,options);

[YPred, probs] = classify(net, augVal);
accuracy = mean(YPred == valImageDS.Labels)

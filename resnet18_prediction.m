net = resnet18();
inputSize = net.Layers(1).InputSize;

img = imread('FlowersSubset/val/daisy/1150395827_6f94a5c6e4_n.jpg');
img = imresize(img, inputSize(1:2));

pred = classify(net, img)

imshow(img);
title(pred);
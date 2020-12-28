%create table from gTruth
DataTable = objectDetectorTrainingData(gTruth);

%shuffle + split
rng(0)
shuffledIndices = randperm(height(DataTable));
idx = floor(0.7 * height(DataTable));

trainingIdx = 1:idx;
trainingDataTbl = DataTable(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = DataTable(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = DataTable(shuffledIndices(testIdx),:);

%create image and label data store
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,2:7));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,2:7));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,2:7));

%combine image and label data store
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);

%display one of the training data
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
%figure
%imshow(annotatedImage)
testData = combine(imdsTest,bldsTest);

%craete Faster R-CNN Network
inputSize = [224 224 3];

%anchor boxes
numAnchors = 3;
boxLabelData = boxLabelDatastore(DataTable(:,2:end));
anchorBoxes = estimateAnchorBoxes(boxLabelData,numAnchors);

featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

numClasses = 6;

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

options = trainingOptions('sgdm',...
    'MaxEpochs',10,...
    'MiniBatchSize',1,...
    'InitialLearnRate',1e-3,...
    'CheckpointPath',tempdir,...
    'ValidationData',validationData);

%[detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
       % 'NegativeOverlapRange',[0 0.3], ...
       % 'PositiveOverlapRange',[0.6 1]);

detector1 = load("detector.mat");
%evaluate detector
detectionResults = detect(detector,testData,'MiniBatchSize',1);
[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);

cp = classperf(testData,detectionResults);
%check the detector
I = imread(testDataTbl.imageFilename{69});
I = imresize(I,inputSize(1:2));
[bboxes,scores,label] = detect(detector,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,(string(label) + ": " + string(scores)));
figure
imshow(I)




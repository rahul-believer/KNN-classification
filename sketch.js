
let video;
const knnClassifier = ml5.KNNClassifier();
let featureExtractor;

function setup() {
  featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
  noCanvas();
  video = createCapture(VIDEO);
  video.parent('videoContainer');
  createButtons();
}

function modelReady() {
  select('#status').html('FeatureExtractor(mobileNet model) Loaded')
}

function addExample(label) {
  const features = featureExtractor.infer(video);

  knnClassifier.addExample(features, label);
  updateCounts();
}

function classify() {
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error('There is no examples in any label');
    return;
  }
  const features = featureExtractor.infer(video);

  knnClassifier.classify(features, gotResults);

}

function createButtons() {
  buttonA = select('#addClassGoForward');
  buttonA.mousePressed(function () {
    addExample('GoForward');
  });

  buttonB = select('#addClassStop');
  buttonB.mousePressed(function () {
    addExample('Stop');
  });

  buttonC = select('#addClassLeft');
  buttonC.mousePressed(function () {
    addExample('Left');
  });

  buttonC = select('#addClassRight');
  buttonC.mousePressed(function () {
    addExample('Right');
  });


  resetBtnA = select('#resetGoForward');
  resetBtnA.mousePressed(function () {
    clearLabel('GoForward');
  });

  resetBtnB = select('#resetStop');
  resetBtnB.mousePressed(function () {
    clearLabel('Stop');
  });

  resetBtnC = select('#resetLeft');
  resetBtnC.mousePressed(function () {
    clearLabel('Left');
  });

  resetBtnD = select('#resetLeft');
  resetBtnD.mousePressed(function () {
    clearLabel('Right');
  });

  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(classify);

  buttonClearAll = select('#clearAll');
  buttonClearAll.mousePressed(clearAllLabels);

  buttonSetData = select('#load');
  buttonSetData.mousePressed(loadMyKNN);

  buttonGetData = select('#save');
  buttonGetData.mousePressed(saveMyKNN);
}

function gotResults(err, result) {
  if (err) {
    console.error(err);
  }

  if (result.confidencesByLabel) {
    const confidences = result.confidencesByLabel;
    if (result.label) {
      select('#result').html(result.label);
      select('#confidence').html(`${confidences[result.label] * 100} %`);
      select('#command').html((result.label == 'GoForward') ? 'Go' : result.label);
    }

    select('#confidenceGoForward').html(`${confidences['GoForward'] ? confidences['GoForward'] * 100 : 0} %`);
    select('#confidenceStop').html(`${confidences['Stop'] ? confidences['Stop'] * 100 : 0} %`);
    select('#confidenceLeft').html(`${confidences['Left'] ? confidences['Left'] * 100 : 0} %`);
    select('#confidenceRight').html(`${confidences['Right'] ? confidences['Right'] * 100 : 0} %`);
  }

  classify();
}

function updateCounts() {
  const counts = knnClassifier.getCountByLabel();

  select('#exampleGoForward').html(counts['GoForward'] || 0);
  select('#exampleStop').html(counts['Stop'] || 0);
  select('#exampleLeft').html(counts['Left'] || 0);
  select('#exampleRight').html(counts['Right'] || 0);
}

function clearLabel(label) {
  knnClassifier.clearLabel(label);
  updateCounts();
}

function clearAllLabels() {
  knnClassifier.clearAllLabels();
  updateCounts();
}

function saveMyKNN() {
  knnClassifier.save('myKNNDataset');
}

function loadMyKNN() {
  knnClassifier.load('./myKNNDataset.json', updateCounts);
}

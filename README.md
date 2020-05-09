# Quantization-Test_bed <br /> 
Quantization of Neural Networks <br /> 

This is a test environment developed for experimenting Quantization of Neural Networks (CNN,MLP). <br /> 



Special thanks to <br /> 

  i) [fchollet](https://github.com/fchollet/deep-learning-models/releases) for providing pretrained models <br />
  ii) [aaron-xichen](https://github.com/aaron-xichen) for providing the preprocessed imagenet validation samples with labels <br />   


Create a folder Data and store the preprocessed imagenet validation images <br />
Create a folder Weights to save all the pretrained models <br />
<br />
We experimented with several methods like <br />

i) Per layer and Per Channel Weight Quantization <br />
ii) Average and Absolute Calibration of Output Activations <br />
iii) Impact of Number of Samples of Calibration Data <br />

Any file can be run by the following command: <br />
<br />
python3 network.py

<br />
The results are saved in Accuracy.txt file


# ArrNet:

The ArrNet model takes in a 30 second seismic waveform and outputs a refined onset time estimate, along with a set of confidence intervals. This is accomplished by utilizing a TCN architecture, trained with quantile loss.

This notebook is a releasable demo, that shows how to instantiate and train the model. The model is trained using the FDSN service to pull waveforms from the internet at each batch, removing the need to store a large dataset locally.


<img src="arrnet_example.png" width="900px">

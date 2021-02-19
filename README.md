# README


## Installation:

Create conda environment with 

	conda create -n CartPoleSimulation python=3.8 matplotlib pyqt pandas tqdm scipy ipython memory_profiler


Choose right installation command for Pytorch
https://pytorch.org/get-started/locally/
(Sensitive to whether you have CUDA (GPU) in your computer or not)

If environment already created:

    conda install matplotlib pyqt pandas tqdm scipy ipython memory_profiler
    pip install do_mpc gekko tensorflow
 
(For Mac without CUDA:)

    conda install pytorch torchvision torchaudio -c pytorch


## Running the project files:

You may run 3 different files: main.py, Train.py and Test.py



Run main.py to start the GUI of CartPole.

Run Train.py to train RNN and Test.py to test it. The data is for both produced on the fly from the CartPole model.

In the “Manual Stabilization” mode you can provide the control input (motor power related to the force acting on the cart) by hovering with your mouse over the lower chart. Due to some bug (or maybe rather specific Cart-Pole system parameters) everything happens to fast to make it doable now.

In the “LQR-Stabilization” mode an automatic (LQR) controller takes care that the Pole stays in the upright position. You can provide the target position of the cart by hovering with your mouse over the lower chart.

Click start button to start the game and stop to stop it.
Quit button is provided because when launch from some IDEs (e.g. Spyder on Windows 10) the standard cross in the window corner may not work correctly.
Generate Trace if clicked in “LQR-Stabilization” mode generates a random target position over time trace and runs a CartPole experiment with LQR controller.

Switch off “Real time simulation” checkbox to make the simulation run faster.

All the experiment recording are saved as .csv data files in the “save” folder.\



Run Train.py to train an RNN to predict future states of a CartPole

Run Test.py to evaluate an RNN predicting future states of a random CartPole experiments

Both Train.py and Test.py generate data from data generated on the fly from random CartPole experiments. You do not have to provide preprepared data.

## Folding convention
Files regions are folded with #region #endregion syntax
For Pycharm default, for Atom install

## Parameter exploration with NNI

For intellegent parameter space exploration with NNI, we have 3 files : 

1. modeling/rnn_tf/Train_nni : Adapted from Train.py to train with parameter exploration
2. modeling/rnn_tf/search_space.json : Search space for parameter search
3. config.yml : Configuring the NNI experiments. 

To run the exploration experiment :

Step 1 : nnictl create --config config.yml
Step 2 : Open the url as displayed on terminal after step 1

## Next step TODO:
Better description of recorded files (controller, git revision, )

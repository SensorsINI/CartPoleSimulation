# README


## Installation:

Get the code from Github: https://github.com/SensorsINI/CartPoleSimulation/tree/Baseline

Create conda environment with 

	conda create -n CartPoleSimulation python=3.8 matplotlib pyqt pandas tqdm scipy gitpython
    pip install do_mpc

n the “Manual Stabilization” mode you can provide the control input (motor power related to the force acting on the cart) by hovering with your mouse over the lower chart. Due to some bug (or maybe rather specific Cart-Pole system parameters) everything happens to fast to make it doable now.

    conda install memory_profiler

Optionally to create gifs:

    conda install imageio

Optionally if you work in Visual Studio Code IDE:
(Don't install otherwise you will get a lot of ugly warnings)

    pip install ptvsd

If environment already created, install packages with:

    conda install matplotlib pyqt pandas tqdm scipy gitpython
    pip install do_mpc

Optionally:

    conda install memory_profiler imageio
    pip install ptvsd
 
Alternatively to set up the environment you can use
requirement.txt (created in maxOS with "conda list -e > requirements.txt")
or
requirements_for_pip.txt (created from requirements.txt with "pip freeze > requirements.txt")
)

## Structure:

The CartPole class in CartPole folder corresponds to a physical cartpole.
It contains methods for setting cartpole parameters, calculating dynamical equation, drawing cartpole and many more.

To perform an experiment CartPole needs an "environment". This environment is provided with CartPole GUI - suitable to visualize dynamical evolution and impact of parameter change -
and Data Generator, with which user can easily generate hours of experimental data. 
  They can be started by running run_cartpole_gui.py and run_data_generator.py respectively.

The CartPole loads controllers from Controllers folder.
One can specify in GUI or in Data Generator which controller should be used.
Adding controllers is easy:
They just have to fit the template provided in Controllers/template_controller.py.
If a file in Controllers folder is named controller_....py and contains the function with the same name (controller_...)
it will be automatically detected and added to the list of possible controllers.

## Operation hints:
  
While the documentation of parameters is still missing in Readme,
they are extensively commented CartPole/cartpole_model.py, GUI/gui_default_params.py and run_data_generator.py
Look inside for more details

In the “Manual Stabilization” mode you can provide the control input (motor power related to the force acting on the cart)
by hovering/clicking with your mouse over the lower chart.
Due current Cart-Pole system parameters everything happens to fast to make it doable now.
Try to change length of the pole and make motor power weaker in cartpole_model.py to make this task feasible.

In the “LQR-Stabilization” mode an automatic (LQR) controller takes care that the Pole stays in the upright position.
You can provide the target position of the cart by hovering or clicking with your mouse over the lower chart.
The same is true for do-mpc controller which is mpc implementation based on true cartpole ODE
done with do-mpc python library

Quit button is provided
because when launched from some IDEs (e.g. Spyder on Windows 10)
the standard cross in the window corner may not work correctly.

The "CSV file name" text box is used for naming both - file to be saved and to be loaded.
If left empty while saving the default name is given.
If left empty while loading data the last (newest) experiment will be loaded

## Folding convention
Files regions are folded with #region #endregion syntax
For Pycharm default, for Atom install

## Parameter exploration with NNI

For intellegent parameter space exploration with NNI, we have 2 special files : 

1. modeling/rnn_tf/search_space.json : Search space for parameter search
2. config.yml : Configuring the NNI experiments. 


Step 1: In Modeling/TF/Train.py comment line:

        train_network()

and uncomment lines:

        nni_parameters = nni.get_next_parameter()
        train_network(nni_parameters)

Step 2 : nnictl create --config Modeling/TF/NNI/config.yml
Step 3 : Open the url as displayed on terminal after Step 1


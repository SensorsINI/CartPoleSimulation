"""
This is a CLASS of predictor.
The idea is to decouple the estimation of system future state from the controller design.
While designing the controller you just chose the predictor you want,
 initialize it while initializing the controller and while stepping the controller you just give it current state
    and it returns the future states

"""

import numpy as np
from SI_Toolkit_ApplicationSpecificFiles.predictors_customization import next_state_predictor_ODE, STATE_VARIABLES


class predictor_ODE:
    def __init__(self, horizon, dt, intermediate_steps=1):

        self.horizon = horizon
        self.batch_size = None  # Will be adjusted to initial input size #TODO: Adjust it to the control size

        self.initial_state = None
        self.output = None

        # Part specific to cartpole
        self.next_step_predictor = next_state_predictor_ODE(dt, intermediate_steps)

    def setup(self, initial_state: np.ndarray):

        # The initial state is provided with not valid second derivatives
        # Batch_size > 1 allows to feed several states at once and obtain predictions parallely
        # Shape of state: (batch size x state variables)

        self.batch_size = np.size(initial_state, 0) if initial_state.ndim > 1 else 1

        # Make sure the input size is at least 2d
        if self.batch_size == 1:
            initial_state = np.expand_dims(initial_state, 0)

        self.initial_state = initial_state

        self.output = np.zeros((self.batch_size, self.horizon + 1, len(STATE_VARIABLES.tolist())), dtype=np.float32)

    def predict(self, Q: np.ndarray, params=None) -> np.ndarray:

        # Shape of Q: (batch size x horizon length)
        if np.size(Q, -1) != self.horizon:
            raise IndexError('Number of provided control inputs does not match the horizon')
        else:
            Q = np.atleast_1d(np.asarray(Q).squeeze())

        if Q.ndim == 1:
            Q = np.expand_dims(Q, 0)

        assert Q.shape[0] == self.initial_state.shape[0]  # Checks ilkf batch size is same for control input and initial_state

        self.output[:, 0, :] = self.initial_state

        for k in range(self.horizon):
            self.output[..., k + 1, :] = self.next_step_predictor.step(self.output[..., k, :], Q[:, k], params)

        return self.output if (self.batch_size > 1) else np.squeeze(self.output)

    def update_internal_state(self, Q0):
        pass


if __name__ == '__main__':
    import timeit
    initialisation = '''
from SI_Toolkit_ApplicationSpecificFiles.predictor_ODE import predictor_ODE
import numpy as np
batch_size = 2000
horizon = 50
predictor = predictor_ODE(horizon, 0.02, 10)
initial_state = np.random.random(size=(batch_size, 6))
Q = np.random.random(size=(batch_size, horizon))
'''


    code = '''\
predictor.setup(initial_state)
predictor.predict(Q)'''

    print(timeit.timeit(code, number=1000, setup=initialisation)/1000.0)

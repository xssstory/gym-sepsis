import gym
from gym.utils import seeding
from tensorflow import keras
import numpy as np
import os
from collections import deque
import pandas as pd
from gym import spaces
import torch

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) 

STATE_MODEL = "model/sepsis_states.model"
TERMINATION_MODEL = "model/sepsis_termination.model"
OUTCOME_MODEL = "model/sepsis_outcome.model"
STARTING_STATES_VALUES = "model/sepsis_starting_states.npz"

NUM_FEATURES = 48  # 46 + action + state index
NUM_ACTIONS = 24

EPISODE_MEMORY = 10


features = ['ALBUMIN', 'ANION GAP', 'BANDS', 'BICARBONATE',
            'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP', 'Glucose',
            'GLUCOSE', 'HeartRate', 'HEMATOCRIT', 'HEMOGLOBIN', 'INR', 'LACTATE',
            'MeanBP', 'PaCO2', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate',
            'SODIUM', 'SpO2', 'SysBP', 'TempC', 'WBC', 'age', 'is_male',
            'race_white', 'race_black', 'race_hispanic', 'race_other', 'height',
            'weight', 'vent', 'sofa', 'lods', 'sirs', 'qsofa', 'qsofa_sysbp_score',
            'qsofa_gcs_score', 'qsofa_resprate_score', 'elixhauser_hospital',
            'blood_culture_positive', 'action', 'state_idx']

class SepsisEnv(gym.Env):
    """
    Built from trained models on top of the MIMIC dataset, this
    Environment simulates the behavior of the Sepsis patient
    in response to medical interventions.
    For details see: https://github.com/akiani/gym-sepsis 
    """
    metadata = {'render.modes': ['ansi']}

    def __init__(self, args, training=True, starting_state=None, verbose=False):
        self.seg_graph = tf.Graph()
        self.sess = tf.Session(graph=self.seg_graph)
        keras.backend.set_session(self.sess)
        with self.seg_graph.as_default():
            module_path = os.path.dirname(__file__)
            seed = args.seed if training else args.seed + 1
            self.verbose = verbose
            self.state_model = keras.models.load_model(os.path.join(module_path, STATE_MODEL))
            self.termination_model = keras.models.load_model(os.path.join(module_path, TERMINATION_MODEL))
            self.outcome_model = keras.models.load_model(os.path.join(module_path, OUTCOME_MODEL))
            self.starting_states = np.load(os.path.join(module_path, STARTING_STATES_VALUES))['sepsis_starting_states']
            self.state_buffer = deque([], maxlen=args.history_length)
            self.window = args.history_length
            self.seed(seed)
            # self.action_space = spaces.Discrete(24)
            self.device = args.device

            # use a pixel to represent next state
            self.observation_space = spaces.Box(low=0, high=NUM_ACTIONS, shape=(NUM_FEATURES-2, 1, 1),
                                                dtype=np.float32)
            self.reset(starting_state=starting_state)
            return

    def step(self, action):
        # create memory of present
        keras.backend.set_session(self.sess)
        with self.seg_graph.as_default():
            self.memory.append(np.append(np.append(self.s.reshape((1, NUM_FEATURES - 2)), action), self.state_idx))
            if self.verbose:
                print("running on memory: ", self.memory)

            memory_array = np.expand_dims(self.memory, 0)
            next_state = self.state_model.predict(memory_array[:, :, :-1])

            # overwrite constant variables (these should't change during episode)
            constants = ['age', 'race_white', 'race_black', 'race_hispanic',
                        'race_other', 'height', 'weight']
            for constant in constants:
                idx = features.index(constant)
                val = self.state_0[idx]
                next_state[0, idx] = val

            termination = self.termination_model.predict(memory_array)
            outcome = self.outcome_model.predict(memory_array)

            termination_categories = ['continue', 'done']
            outcome_categories = ['death', 'release']

            termination_state = termination_categories[np.argmax(termination)]
            outcome_state = outcome_categories[np.argmax(outcome)]

            reward = 0
            done = False

            if termination_state == 'done':
                done = True
                if outcome_state == 'death':
                    reward = -15
                else:
                    reward = 15

            # keep next state in memory
            self.s = next_state.reshape(46, 1, 1)
            self.state_idx += 1
            self.rewards.append(reward)
            self.dones.append(done)
            self.state_buffer.append(torch.tensor(self.s, device=self.device, dtype=torch.float32))
            return torch.stack(list(self.state_buffer), 0), reward, done, {"prob" : 1}

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(NUM_FEATURES-2, 1, 1, device=self.device))

    def reset(self, starting_state=None):
        self.rewards = []
        self.dones = []
        self.state_idx = 0
        self.memory = deque([np.zeros(shape=[NUM_FEATURES])] * 10, maxlen=10)
        self._reset_buffer()
        if starting_state is None:
            self.s = self.starting_states[self.rng.randint(0, len(self.starting_states))][:-1]
        else:
            self.s = starting_state

        self.s = self.s.reshape(NUM_FEATURES - 2, 1, 1)
        self.state_0 = np.copy(self.s)

        if self.verbose:
            print("starting state:", self.s)
        self.state_buffer.append(torch.tensor(self.s, device=self.device, dtype=torch.float32))
        return torch.stack(list(self.state_buffer), 0)

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='ansi'):
        df = pd.DataFrame(self.memory, columns=features, index=range(0, 10))
        print(df)
    
    def action_space(self):
        return NUM_ACTIONS

    def close(self):
        keras.backend.set_session(self.sess)
        keras.backend.clear_session()
        del self.state_model
        del self.termination_model
        del self.outcome_model
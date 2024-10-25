"""Policy class containing hand crafted policies for various WOFOST Gym Environments

Written by: Will Solow, 2024
"""
import gymnasium as gym
from wofost_gym.exceptions import PolicyException
from wofost_gym.envs.wofost_base import Plant_NPK_Env, Harvest_NPK_Env
from abc import abstractmethod

class Policy:
    """Abstract Policy class containing the validate method

    Requires that the Gym Environment is wrapped with the NPKDiscreteObservationWrapper
    to ensure easy policy specification, specifically for policies that depend on
    the state. 
    """
    required_vars = list

    def __init__(self, env: gym.Env, required_vars: list=[]):
        """Initialize the :class:`Policy`.

        Args: 
            env: The Gymnasium Environment
            required_vars: list of required state space variables 
        """
        self.required_vars = required_vars
        self.env = env

        self._validate()
    
    def __call__(self, obs:dict):
        """Calls the _get_action() method. 
        
        Helper method so that the HandCrafted Policies share same call 
        structure to RL Agents. So, RL Agent Policy or Handcrafted Policy can be 
        returned from a function and called even without knowing what type of policy 
        it is."""
        return self._get_action(obs)
    
    def _validate(self):
        """Check that the policy is valid given the observation space and that
        the environment is wrapped with the NPKDictObservationWrapper
        """
        obs = self.env.observation_space.sample()
        if isinstance(obs, dict):
            for key in self.required_vars:
                if not key in list(obs.keys()):
                    msg = f"Required observation `{key}` for policy is not in inputted observation "
                    raise PolicyException(msg)
        else:
            msg = "Observation Space is not of type `Dict`. Wrap Environment with NPKDictObservationWrapper before proceeding."
            raise PolicyException(msg)
        
        action = self.env.action_space.sample()

        if not isinstance(action, dict):
            msg = "Action Space is not of type `Dict`. Wrap Environment with NPKDictActionWrapper before proceeding."
            raise PolicyException(msg)

    @abstractmethod     
    def _get_action(self, obs:dict):
        """Return the action for the environment to take
        """
        msg = "Policy Subclass should implement this"
        raise NotImplementedError(msg) 
    
    @abstractmethod
    def __str__(self):
        """
        Returns the string representation
        """

class No_Action(Policy):
    """Default policy performing no irrigation or fertilization actions
    """
    required_vars = []

    def __init__(self, env: gym.Env):
        """Initialize the :class:`No_Action`.

        Args: 
            env: The Gymnasium Environment
        """
        super().__init__(env, required_vars=self.required_vars)

    def _get_action(self, obs):
        return {'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
    
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"no_action"
    
class Weekly_N(Policy):
    """Policy applying a small amount of Nitrogen every day
    """
    required_vars = []

    def __init__(self, env: gym.Env, amount: int=0):
        """Initialize the :class:`Weekly_N`.

        Args: 
            env: The Gymnasium Environment
            required_vars: list of required state space variables 
        """
        self.amount = amount
        super().__init__(env, required_vars=self.required_vars)
        

    def _validate(self):
        """Validates that the weekly amount is within the range of allowable actions
        """
        super()._validate()

        if self.amount > self.env.num_fert:
            msg = "N Amount exceeds total Nitrogen actions"
            raise PolicyException(msg)
        

    def _get_action(self, obs:dict):
        """Return an action with an amount of N fertilization
        """
        return {'n': self.amount, 'p': 0, 'k': 0, 'irrig':0 }
    
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"weeklyn_{self.amount}"
    
class Interval_N(Policy):
    """Policy applying a small amount of Nitrogen at a given interval
    """
    required_vars = ["DAYS"]

    def __init__(self, env: gym.Env, amount: int=0, interval: int=1):
        """Initialize the :class:`Interval_N`.

        Args: 
            env: The Gymnasium Environment
            required_vars: list of required state space variables 
        """
        self.amount = amount
        self.interval = interval
        super().__init__(env, required_vars=self.required_vars)
        

    def _validate(self):
        """Validates that the weekly amount is within the range of allowable actions
        """
        super()._validate()

        if self.amount > self.env.num_fert:
            msg = "W Amount exceeds total Fertilization actions"
            raise PolicyException(msg)
        

    def _get_action(self, obs:dict):
        """Return an action with an amount of N fertilization
        """
        if obs["DAYS"] % self.interval == 0:
            return {'n': self.amount, 'p': 0, 'k': 0, 'irrig':0 }
        else:
            return {'n': 0, 'p': 0, 'k': 0, 'irrig': 0}
    
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"intervaln_{self.interval}_{self.amount}"
        
class Interval_W(Policy):
    """Policy applying a small amount of Water at a given interval
    """
    required_vars = ["DAYS"]

    def __init__(self, env: gym.Env, amount: int=0, interval: int=1):
        """Initialize the :class:`Interval_W`.

        Args: 
            env: The Gymnasium Environment
            required_vars: list of required state space variables 
        """
        self.amount = amount
        self.interval = interval
        super().__init__(env, required_vars=self.required_vars)
        

    def _validate(self):
        """Validates that the weekly amount is within the range of allowable actions
        """
        super()._validate()

        if self.amount > self.env.num_irrig:
            msg = "W Amount exceeds total Irrigation actions"
            raise PolicyException(msg)
        

    def _get_action(self, obs:dict):
        """Return an action with an amount of N fertilization
        """
        if obs["DAYS"] % self.interval == 0:
            return {'n': 0, 'p': 0, 'k': 0, 'irrig':self.amount }
        else:
            return {'n': 0, 'p': 0, 'k': 0, 'irrig': 0}
        
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"intervalw_{self.interval}_{self.amount}"


class No_Action_Harvest(Policy):
    """Default policy for performing no irrigation or fertilization actions
    in a Harvest Environment
    """
    required_vars = ["DAYS"]

    def __init__(self, env:gym.Env):
        """Initialize the :class:`No_Action_Harvest`.

        Args: 
            env: The Gymnasium Environment
            required_vars: list of required state space variables 
        """
        super().__init__(env, required_vars=self.required_vars)

    def _validate(self):
        """Validates that the environment is a harvesting environment
        """
        super()._validate()

        if not isinstance(self.env.unwrapped, Harvest_NPK_Env):
            msg = "Environment does not inherit from `Harvest_NPK_Env`"
            raise PolicyException(msg)
        
    def _get_action(self, obs:dict):
        """Return an action which harvests on day 225
        """
        if obs["DAYS"] == 225:
            return {'harvest': 1, 'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
        
        return {'harvest': 0, 'n': 0, 'p': 0, 'k': 0, 'irrig':0 }

    def __str__(self):
        """
        Returns a human readable string
        """
        return f"noactionharvest"
    
class No_Action_Plant(Policy):
    """Default policy for performing no irrigation or fertilization actions
    in a Plant Environment
    """
    required_vars = ["DAYS"]

    def __init__(self, env:gym.Env):
        """Initialize the :class:`No_Action_Plant`.

        Args: 
            env: The Gymnasium Environment
            required_vars: list of required state space variables 
        """
        super().__init__(env, required_vars=self.required_vars)

    def _validate(self):
        """Validates that the environment is a planting environment
        """
        super()._validate()

        if not isinstance(self.env.unwrapped, Plant_NPK_Env):
            msg = "Environment does not inherit from `Plant_NPK_Env`"
            raise PolicyException(msg)
        
    def _get_action(self, obs:dict):
        """Return an action which plants on day 30 and harvests on day 225
        """
        if obs["DAYS"] == 30:
            return {'plant': 1, 'harvest': 0, 'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
        if obs["DAYS"] == 225:
            return {'plant': 0, 'harvest': 1, 'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
        
        return {'plant': 0, 'harvest': 0, 'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
    
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"noactionplant"

class Threshold_N(Policy):
    required_vars = ['TOTN']
    def __init__(self, env: gym.Env, threshold:float=10, amount:int=1):
        """Initialize the :class:`Threshold_N`.

        Args: 
            env: The Gymnasium Environment
            threshold: the amount of Nitrogen that can be applied
            amount: the amount of Nitrogen to be applied
        """
        self.threshold = threshold
        self.amount = amount
        super().__init__(env, required_vars=self.required_vars)
    
    def _get_action(self, obs):
        """Returns an action that applies Nitrogen until a threshold is met
        """
        if obs['TOTN'] > self.threshold:
            return {'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
        else:
            return {'n': self.amount, 'p': 0, 'k': 0, 'irrig':0 }
        
    def _validate(self):
        """Validates that the weekly amount is within the range of allowable actions
        """
        super()._validate()

        if self.amount > self.env.num_fert:
            msg = "N Amount exceeds total Nitrogen actions"
            raise PolicyException(msg)
        
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"thresholdn_{self.threshold}_{self.amount}"
        
class Below_N(Policy):
    required_vars = ['NAVAIL']

    def __init__(self, env: gym.Env, threshold:float=1, amount:int=1):
        """Initialize the :class:`Below_N`.

        Args: 
            env: The Gymnasium Environment
            threshold: the threshold below which to apply nitrogen
            amount: the amount of Nitrogen to be applied
        """
        self.threshold = threshold
        self.amount = amount
        super().__init__(env, required_vars=self.required_vars)

    def _get_action(self, obs):
        """Returns an action that applies Nitrogen while below a certain threshold
        """
        if obs['NAVAIL'] < self.threshold:
            return {'n': self.amount, 'p': 0, 'k': 0, 'irrig':0 }
        else:
            return {'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
        
    def _validate(self):
        """Validates that the weekly amount is within the range of allowable actions
        """
        super()._validate()

        if self.amount > self.env.num_fert:
            msg = "N Amount exceeds total Nitrogen actions"
            raise PolicyException(msg)

    def __str__(self):
        """
        Returns a human readable string
        """
        return f"belown_{self.threshold}_{self.amount}"
        
class Below_I(Policy):
    required_vars = ['SM']

    def __init__(self, env: gym.Env, threshold:float=1, amount:int=1):
        """Initialize the :class:`Below_I`.

        Args: 
            env: The Gymnasium Environment
            threshold: the threshold below which to apply irrigation
            amount: the amount of Irrigation to be applied
        """
        self.threshold = threshold
        self.amount = amount
        super().__init__(env, required_vars=self.required_vars)

    def _get_action(self, obs):
        """Returns an action that applies Nitrogen while below a certain threshold
        """
        if obs['SM'] < self.threshold:
            return {'n': 0, 'p': 0, 'k': 0, 'irrig':self.amount }
        else:
            return {'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
        
    def _validate(self):
        """Validates that the weekly amount is within the range of allowable actions
        """
        super()._validate()

        if self.amount > self.env.num_irrig:
            msg = "N Amount exceeds total Irrigation actions"
            raise PolicyException(msg)
        
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"belowi_{self.threshold}_{self.amount}"
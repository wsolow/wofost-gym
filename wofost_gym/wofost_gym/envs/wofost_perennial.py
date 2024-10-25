"""Main API for default WOFOST Gym environments with actions for NPK and water
application.

Used for perennial crop simulations.
"""

import gymnasium as gym

from wofost_gym.args import NPK_Args
from wofost_gym import utils
from wofost_gym.envs.wofost_base import NPK_Env

import pcse
from pcse.soil.soil_wrappers import SoilModuleWrapper_LNPKW
from pcse.soil.soil_wrappers import SoilModuleWrapper_LN
from pcse.soil.soil_wrappers import SoilModuleWrapper_LNPK
from pcse.soil.soil_wrappers import SoilModuleWrapper_PP
from pcse.soil.soil_wrappers import SoilModuleWrapper_LW
from pcse.soil.soil_wrappers import SoilModuleWrapper_LNW
from pcse.crop.wofost8 import Wofost80Perennial
from pcse.agromanager import AgroManagerPerennial

class Perennial_Limited_NPKW_Env(NPK_Env):
    """Simulates crop growth under NPK and water limited conditions
    """
    config = utils.make_config(soil=SoilModuleWrapper_LNPKW, crop=Wofost80Perennial, \
                               agro=AgroManagerPerennial)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Limited_NPKW_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(1)
    
    def _take_action(self, action: int):
        """Controls sending fertilization and irrigation signals to the model. 
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        n_amount = 0
        p_amount = 0
        k_amount = 0
        irrig_amount = 0

        # Null action
        if action == 0: 
            return (n_amount, p_amount, k_amount, irrig_amount)
        
        # Irrigation action
        if action >= 3 * self.num_fert+1:
            i_amount = action - (3 * self.num_fert)
            i_amount *= self.irrig_amount
            self.model._send_signal(signal=pcse.signals.irrigate, amount=i_amount, \
                                    efficiency=self.irrig_effec)
            return (n_amount, p_amount, k_amount, irrig_amount)
        
        # Fertilizaiton action, correct for null action
        if (action-1) // self.num_fert == 0:
            n_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                    N_amount=n_amount, N_recovery=self.n_recovery)
        elif (action-1) // self.num_fert == 1:
            p_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                    P_amount=p_amount, P_recovery=self.p_recovery)
        elif (action-1) // self.num_fert == 2:
            k_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        K_amount=k_amount, K_recovery=self.k_recovery)
            
        return (n_amount, p_amount, k_amount, irrig_amount)

class Perennial_PP_Env(NPK_Env):
    """Simulates Potential Production. That is how much the crop would grow
    with abundant NPK/Water
    """
    config = utils.make_config(soil=SoilModuleWrapper_PP, crop=Wofost80Perennial, \
                               agro=AgroManagerPerennial)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`PP_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(1)

    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        No actions available in this Potential Production Env 
        
        Args:
            action
        """
        return (0, 0, 0, 0)

class Perennial_Limited_NPK_Env(NPK_Env):
    """Simulates crop growth under NPK Limited Production 
    """
    config = utils.make_config(soil=SoilModuleWrapper_LNPK, crop=Wofost80Perennial, \
                               agro=AgroManagerPerennial)

    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Limited_NPK_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(1+3*self.num_fert)

    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        n_amount = 0
        p_amount = 0
        k_amount = 0

        # Null action
        if action == 0: 
            return (n_amount, p_amount, k_amount, 0)
        
        # Fertilizaiton action, correct for null action
        if (action-1) // self.num_fert == 0:
            n_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                    N_amount=n_amount, N_recovery=self.n_recovery)
        elif (action-1) // self.num_fert == 1:
            p_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                    P_amount=p_amount, P_recovery=self.p_recovery)
        elif (action-1) // self.num_fert == 2:
            k_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        K_amount=k_amount, K_recovery=self.k_recovery)
            
        return (n_amount, p_amount, k_amount, 0)

class Perennial_Limited_N_Env(NPK_Env):
    """Simulates crop growth under Nitrogen Limited Production 
    """
    config = utils.make_config(soil=SoilModuleWrapper_LN, crop=Wofost80Perennial, \
                               agro=AgroManagerPerennial)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Limited_N_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(1+self.num_fert)


    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        n_amount = 0

        # Null action
        if action == 0: 
            return (n_amount, 0, 0, 0)
        
        # Fertilizaiton action, correct for null action
        if (action-1) // self.num_fert == 0:
            n_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                    N_amount=n_amount, N_recovery=self.n_recovery)
            
        return (n_amount, 0, 0, 0)

class Perennial_Limited_NW_Env(NPK_Env):
    """Simulates crop growth under Nitrogen and Water Limited Production 
    """
    config = utils.make_config(soil=SoilModuleWrapper_LNW, crop=Wofost80Perennial, \
                               agro=AgroManagerPerennial)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Limited_NW_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(1+self.num_fert + self.num_irrig)

    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        n_amount = 0
        irrig_amount = 0

        # Null action
        if action == 0: 
            return (n_amount, 0, 0, irrig_amount)
        
        # Irrigation action
        if action >= self.num_fert+1:
            i_amount = action - (self.num_fert)
            i_amount *= self.irrig_amount
            self.model._send_signal(signal=pcse.signals.irrigate, amount=i_amount, \
                                    efficiency=self.irrig_effec)
            return (n_amount, 0, 0, irrig_amount)
        
        # Fertilizaiton action, correct for null action
        if (action-1) // self.num_fert == 0:
            n_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=pcse.signals.apply_npk, \
                                    N_amount=n_amount, N_recovery=self.n_recovery)
            
        return (n_amount, 0, 0, irrig_amount)

class Perennial_Limited_W_Env(NPK_Env):
    """Simulates crop growth under Water Limited Production 
    """
    config = utils.make_config(soil=SoilModuleWrapper_LW, crop=Wofost80Perennial, \
                               agro=AgroManagerPerennial)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Limited_W_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(1+self.num_irrig)

    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        irrig_amount = action
        # Null action
        if action == 0: 
            return (0, 0, 0, irrig_amount)
        
        # Irrigation action
        if action >= 0 * self.num_fert+1:
            i_amount = action - (0 * self.num_fert)
            i_amount *= self.irrig_amount
            self.model._send_signal(signal=pcse.signals.irrigate, amount=i_amount, \
                                    efficiency=self.irrig_effec)

        return (0, 0, 0, irrig_amount)
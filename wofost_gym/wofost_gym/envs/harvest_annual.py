"""Main API for simulating crop growth NPK fertilization and irrigation actions
and the inclusion of planting and harvesting of the crop.

Used for single year annual crop simulations.
"""
import gymnasium as gym

from wofost_gym import utils
from wofost_gym.args import NPK_Args
from wofost_gym.envs.wofost_base import Harvest_NPK_Env

import pcse
from pcse.soil.soil_wrappers import SoilModuleWrapper_LNPKW
from pcse.soil.soil_wrappers import SoilModuleWrapper_LN
from pcse.soil.soil_wrappers import SoilModuleWrapper_LNPK
from pcse.soil.soil_wrappers import SoilModuleWrapper_PP
from pcse.soil.soil_wrappers import SoilModuleWrapper_LW
from pcse.soil.soil_wrappers import SoilModuleWrapper_LNW
from pcse.crop.wofost8 import Wofost80
from pcse.agromanager import AgroManagerHarvest


class Harvest_Limited_NPKW_Env(Harvest_NPK_Env):
    """Simulates crop growth under NPK and Water Limited Production 
    with action for harvesting
    """
    config = utils.make_config(soil=SoilModuleWrapper_LNPKW, crop=Wofost80, \
                               agro=AgroManagerHarvest)

    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Harvest_Limited_NPKW_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(2+3*args.num_fert+args.num_irrig)

    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Includes action for harvesting.
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        h_act = 0
        n_amount = 0
        p_amount = 0
        k_amount = 0
        irrig_amount = 0

        # Null Action
        if action == 0:
            return (h_act, n_amount, p_amount, k_amount, irrig_amount)
        
        # Harvesting Action
        elif action == 1:
            if self.active_crop_flag:
                self.model._send_signal(signal=pcse.signals.crop_harvest, day=self.date,\
                                        effiency=self.Harvest_effec)
            h_act=1
            return ((h_act, n_amount, p_amount, k_amount, irrig_amount))

         # Irrigation action
        if action >= 3 * self.num_fert+2:
            i_amount = action - (3 * self.num_fert)-1
            i_amount *= self.irrig_amount
            self.model._send_signal(signal=pcse.signals.irrigate, amount=i_amount, \
                                    efficiency=self.irrig_effec)
            return (h_act, n_amount, p_amount, k_amount, irrig_amount)
        
        # Fertilizaiton action, correct for 2 crop specific actions (harvest/plant) and null action
        if action >= 3:
            action -= 2
            # Nitrogen fertilization
            if action // self.num_fert == 0:
                n_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        N_amount=n_amount, N_recovery=self.n_recovery)
            elif action // self.num_fert == 1:
                p_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        P_amount=p_amount, P_recovery=self.p_recovery)
            elif action // self.num_fert == 2:
                k_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        K_amount=k_amount, K_recovery=self.k_recovery)
                
        return ((h_act, n_amount, p_amount, k_amount, irrig_amount))

class Harvest_PP_Env(Harvest_NPK_Env):
    """Simulates crop growth under abundant NPK and water
    with action for harvesting
    """
    config = utils.make_config(soil=SoilModuleWrapper_PP, crop=Wofost80, \
                               agro=AgroManagerHarvest)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Harvest_PP_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)
        
        self.action_space = gym.spaces.Discrete(2)

      
    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Includes action for Planting.
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        h_act = 0

        # Null Action
        if action == 0:
            return (h_act, 0, 0, 0, 0)
        
        # Planting Action
        elif action == 1:
            if self.active_crop_flag:
                self.model._send_signal(signal=pcse.signals.crop_harvest, day=self.date,\
                                        effiency=self.Harvest_effec)
            h_act=1
            return (h_act, 0, 0, 0, 0)

class Harvest_Limited_NPK_Env(Harvest_NPK_Env):
    """Simulates crop growth under NPK Limited Production 
    with action for harvesting
    """
    config = utils.make_config(soil=SoilModuleWrapper_LNPK, crop=Wofost80, \
                               agro=AgroManagerHarvest)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Harvest_Limited_NPK_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(2+3*self.num_fert)

      
    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Includes action for harvesting.
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        h_act = 0
        n_amount = 0
        p_amount = 0
        k_amount = 0

        # Null Action
        if action == 0:
            return (h_act, n_amount, p_amount, k_amount, 0)
    
        # Harvesting Action
        elif action == 1:
            if self.active_crop_flag:
                self.model._send_signal(signal=pcse.signals.crop_harvest, day=self.date,\
                                        effiency=self.Harvest_effec)
            h_act=1
            return (h_act, n_amount, p_amount, k_amount, 0)

        # Fertilizaiton action, correct for 2 crop specific actions (harvest/plant) and null action
        if action >= 2:
            action -= 2
            # Nitrogen fertilization
            if action // self.num_fert == 0:
                n_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        N_amount=n_amount, N_recovery=self.n_recovery)
            elif action // self.num_fert == 1:
                p_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        P_amount=p_amount, P_recovery=self.p_recovery)
            elif action // self.num_fert == 2:
                k_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        K_amount=k_amount, K_recovery=self.k_recovery)
                
        return (h_act, n_amount, p_amount, k_amount, 0)
        
class Harvest_Limited_N_Env(Harvest_NPK_Env):
    """Simulates crop growth under Nitrogen Limited Production 
    with action for harvesting
    """
    config = utils.make_config(soil=SoilModuleWrapper_LN, crop=Wofost80, \
                               agro=AgroManagerHarvest)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Harvest_Limited_N_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)
        self.action_space = gym.spaces.Discrete(2+self.num_fert)

    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Includes action for harvesting.
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        h_act = 0
        n_amount = 0

        # Null Action
        if action == 0:
            return (h_act, n_amount, 0, 0, 0)
    
        # Harvesting Action
        elif action == 1:
            if self.active_crop_flag:
                self.model._send_signal(signal=pcse.signals.crop_harvest, day=self.date,\
                                        effiency=self.Harvest_effec)
            h_act=1
            return (h_act, n_amount, 0, 0, 0)

        
        # Fertilizaiton action, correct for 2 crop specific actions (harvest/plant) and null action
        if action >= 2:
            action -= 2
            # Nitrogen fertilization
            if action // self.num_fert == 0:
                n_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        N_amount=n_amount, N_recovery=self.n_recovery)
                
        return (h_act, n_amount, 0, 0, 0)
 
class Harvest_Limited_NW_Env(Harvest_NPK_Env):
    """Simulates crop growth under Nitrogen and Water Limited Production 
    with action for harvesting
    """
    config = utils.make_config(soil=SoilModuleWrapper_LNW, crop=Wofost80, \
                               agro=AgroManagerHarvest)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Harvest_Limited_NW_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)
        self.action_space = gym.spaces.Discrete(2+self.num_fert + self.num_irrig)

      
    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Includes action for harvesting.
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        h_act = 0
        n_amount = 0
        irrig_amount = 0

        # Null Action
        if action == 0:
            return (h_act, n_amount, 0, 0, irrig_amount)
        
        # Harvesting Action
        elif action == 1:
            if self.active_crop_flag:
                self.model._send_signal(signal=pcse.signals.crop_harvest, day=self.date,\
                                        effiency=self.Harvest_effec)
            h_act=1
            return (h_act, n_amount, 0, 0, irrig_amount)

         # Irrigation action
        if action >= 1 * self.num_fert+2:
            i_amount = action - (1 * self.num_fert) - 1
            i_amount *= self.irrig_amount
            self.model._send_signal(signal=pcse.signals.irrigate, amount=i_amount, \
                                    efficiency=self.irrig_effec)
            return (h_act, n_amount, 0, 0, irrig_amount)
        
        # Fertilizaiton action, correct for 2 crop specific actions (harvest/plant) and null action
        if action >= 2:
            action -= 2
            # Nitrogen fertilization
            if action // self.num_fert == 0:
                n_amount = self.fert_amount * ((action % self.num_fert) + 1) 
                self.model._send_signal(signal=pcse.signals.apply_npk, \
                                        N_amount=n_amount, N_recovery=self.n_recovery)
                
        return (h_act, n_amount, 0, 0, irrig_amount)

class Harvest_Limited_W_Env(Harvest_NPK_Env):
    """Simulates crop growth under Water Limited Production 
    with action for harvesting
    """
    config = utils.make_config(soil=SoilModuleWrapper_LW, crop=Wofost80, \
                               agro=AgroManagerHarvest)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Harvest_Limited_W_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(2+self.num_irrig)
 
    def _take_action(self, action:int):
        """Controls sending fertilization and irrigation signals to the model. 
        Includes action for harvesting.
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        h_act = 0
        irrig_amount = 0

        # Null Action
        if action == 0:
            return (h_act, 0, 0, 0, irrig_amount)
        
        # Harvesting Action
        elif action == 1:
            if self.active_crop_flag:
                self.model._send_signal(signal=pcse.signals.crop_harvest, day=self.date,\
                                        effiency=self.Harvest_effec)
            h_act=1
            return (h_act, 0, 0, 0, irrig_amount)

         # Irrigation action
        if action >= 0 * self.num_fert+2:
            i_amount = action - (0 * self.num_fert) - 1
            i_amount *= self.irrig_amount
            self.model._send_signal(signal=pcse.signals.irrigate, amount=i_amount, \
                                    efficiency=self.irrig_effec)
            return (h_act, 0, 0, 0, irrig_amount)
        

                
        return (h_act, 0, 0, 0, irrig_amount)
   

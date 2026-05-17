from xuance.environment.utils import EnvironmentDict
from xuance.common import Optional
from xuance.environment.multi_agent_env.mpe import MPE_Env

REGISTRY_MULTI_AGENT_ENV: Optional[EnvironmentDict] = {
    "mpe": MPE_Env,
}

try:
    from xuance.environment.multi_agent_env.drones import Drones_MultiAgentEnv
    REGISTRY_MULTI_AGENT_ENV['Drones'] = Drones_MultiAgentEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["Drones"] = str(error)

try:
    from xuance.environment.multi_agent_env.football import GFootball_Env
    REGISTRY_MULTI_AGENT_ENV['Football'] = GFootball_Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["Football"] = str(error)

try:
    from xuance.environment.multi_agent_env.robotic_warehouse import RoboticWarehouseEnv
    REGISTRY_MULTI_AGENT_ENV['RoboticWarehouse'] = RoboticWarehouseEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["RoboticWarehouse"] = str(error)

try:
    from xuance.environment.multi_agent_env.starcraft2 import StarCraft2_Env
    REGISTRY_MULTI_AGENT_ENV['StarCraft2'] = StarCraft2_Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["StarCraft2"] = str(error)

try:
    from xuance.environment.multi_agent_env.atari import AtariMultiAgentEnv
    REGISTRY_MULTI_AGENT_ENV['atari'] = AtariMultiAgentEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["atari"] = str(error)

try:
    from xuance.environment.multi_agent_env.uav_pursuit_cel_maddpg import UAVPursuitCelMaddpgEnv
    REGISTRY_MULTI_AGENT_ENV['uav_pursuit_cel_maddpg'] = UAVPursuitCelMaddpgEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["uav_pursuit_cel_maddpg"] = str(error)

try:
    from xuance.environment.multi_agent_env.uav_pursuit_apollonius_obs import UAVPursuitApolloniusObsEnv
    REGISTRY_MULTI_AGENT_ENV['uav_pursuit_apollonius_obs'] = UAVPursuitApolloniusObsEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["uav_pursuit_apollonius_obs"] = str(error)

try:
    from xuance.environment.multi_agent_env.uav_pursuit_apollonius_obs_2 import UAVPursuitApolloniusObs2Env
    REGISTRY_MULTI_AGENT_ENV['uav_pursuit_apollonius_obs_2'] = UAVPursuitApolloniusObs2Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["uav_pursuit_apollonius_obs_2"] = str(error)

try:
    from xuance.environment.multi_agent_env.uav_pursuit_apollonius_obs_3 import UAVPursuitApolloniusObs3Env
    REGISTRY_MULTI_AGENT_ENV['uav_pursuit_apollonius_obs_3'] = UAVPursuitApolloniusObs3Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["uav_pursuit_apollonius_obs_3"] = str(error)
    

try:
    from xuance.environment.multi_agent_env.uav_pursuit_apollonius_obs_4 import UAVPursuitApolloniusObs4Env
    REGISTRY_MULTI_AGENT_ENV['uav_pursuit_apollonius_obs_4'] = UAVPursuitApolloniusObs4Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["uav_pursuit_apollonius_obs_4"] = str(error)

try:
    from xuance.environment.multi_agent_env.uav_pursuit_apollonius_obs_5 import UAVPursuitApolloniusObs5Env
    REGISTRY_MULTI_AGENT_ENV['uav_pursuit_apollonius_obs_5'] = UAVPursuitApolloniusObs5Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["uav_pursuit_apollonius_obs_5"] = str(error)
__all__ = [
    "REGISTRY_MULTI_AGENT_ENV",
]

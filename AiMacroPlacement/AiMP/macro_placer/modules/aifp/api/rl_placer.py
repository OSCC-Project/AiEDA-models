from aimp.aifp.operation.macro_placer.rl_placer.rl_placer import RLPlacer
from aimp.aifp import setting

def run_rl_placer(*args):
    rl_placer = RLPlacer()
    rl_placer.run(setting.rl_config['device'])
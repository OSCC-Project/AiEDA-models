from macro_placer.modules.safp.sa_placer import SAPlacer
from tools.iEDA.module.placement import IEDAPlacement

class RunSAPlacer:
    def __init__(self, ieda_placer : IEDAPlacement):
        self.ieda_placer = ieda_placer
    
    def run(self,config : str,  tcl_path = ""):
        
        self.ieda_placer.run_mp(config, tcl_path)
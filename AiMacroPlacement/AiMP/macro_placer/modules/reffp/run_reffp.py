from tools.iEDA.module.placement import IEDAPlacement

class RunRefPlacer:
    def __init__(self, ieda_placer : IEDAPlacement):
        self.ieda_placer = ieda_placer
    
    def run(self, tcl_path = ""):
        
        self.ieda_placer.run_refinement(tcl_path)
from serve_navigation_quant_timing import main
from navigation_stage2_common import ROOT
from lmdeploy.pytorch.models.patch import update_custom_module_map
from navigation_quant_attribution import instrument_frontend
update_custom_module_map(str(ROOT/'checks/navigation_quant_attribution_module_map.py'))
instrument_frontend()
if __name__=='__main__':
    main()

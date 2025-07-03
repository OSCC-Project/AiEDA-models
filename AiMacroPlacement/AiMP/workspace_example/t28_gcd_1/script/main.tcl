
################################################################################
#  tcl main
################################################################################
source /data/project_share/dataset_baseline/gcd/workspace/script/definition.tcl

if {[string equal $STEP "full_flow"]} {
    if {[string equal $PROCESS_NODE "ng45"]} {
        source ${SRC_SCRIPT}/${EDA_TOOL}_${PROCESS_NODE}.tcl
    }
    if {[string equal $PROCESS_NODE "t28"]} {
        source ${SRC_SCRIPT}/${EDA_TOOL}_mp.tcl
    }
} elseif {[string equal $STEP "drc"]} {
     source ${SRC_SCRIPT}/${EDA_TOOL}_drc.tcl
} else {
    if {[string equal $PROCESS_NODE "t28"]} {
        source ${SRC_SCRIPT}/${EDA_TOOL}.tcl
    }
}

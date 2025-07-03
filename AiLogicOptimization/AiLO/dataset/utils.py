# Open Core dataset
# design1 = ['i2c','spi','des3_area','ss_pcm','usb_phy','sasc','wb_dma','simple_spi','dynamic_node']
# design2 = ['aes','pci','ac97_ctrl','mem_ctrl','tv80','fpu']
# design3 = ['wb_conmax','tinyRocket','aes_xcrypt','aes_secworks']
# design4 = ['jpeg','bp_be','ethernet','vga_lcd','picosoc']
# design5 = ['dft','idft','fir','iir','sha256']

# designs = design1 + design2 + design3 + design4 +design5

# EPFL dataset
design1= ['adder','bar','max','sin','i2c', 'cavlc','ctrl','int2float','priority','router']
design2 = ['div','log2','multiplier', 'sqrt','square','arbiter','mem_ctrl','voter', 'hyp']
designs = design1 + design2

node_types = {
    "pi" : 0,
    "po" : 1,
    "and" : 2,
    "not" : 3,
    "buf" : 4
}

edge_types = {
    "not" : 0,
    "buf" : 1,
}

OptDict = {
    1: "refactor",
    2: "refactor -z",
    3: "refactor -l",
    4: "refactor -l -z",
    5: "rewrite",
    6: "rewrite -z",
    7: "rewrite -l",
    8: "rewrite -l -z" ,
    9: "resub",
    10: "resub -z",
    11: "resub -l",
    12: "resub -l -z",
    13: "balance"
}

OptDict_reverse = {v: k for k, v in OptDict.items()}
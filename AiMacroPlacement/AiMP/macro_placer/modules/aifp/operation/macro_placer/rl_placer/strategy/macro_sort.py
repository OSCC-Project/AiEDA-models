class MacroSort:
    def __init__(self):
        pass
    
    def sort_area_desc(unfixed_macro_indices:list, inst_list:list):
        unfixed_macro_indices.sort(key=lambda macro_idx: inst_list[macro_idx].get_width() * inst_list[macro_idx].get_height(), reverse=True)
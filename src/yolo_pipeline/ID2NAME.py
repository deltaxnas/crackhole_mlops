class MAP:
    def __init__(self):
        self.panel_id_to_name = {
            "P24_S0": "FENDER_SP3k_0",
            "P23_S0": "FENDER_CV_0",
            "P6_S0": "FENDER_PE_MQ4_0",
            "P22_S0": "FENDER_SG2_0",
            "P22_S17": "FENDER_SG2_17",
            "P22_S32": "FENDER_SG2_32",
            "P22_S33": "FENDER_SG2_33",
            "P8_S0": "FENDER_SW_0",
            "P7_S0": "FENDER_TK_0",
            "P7_S1": "FENDER_TK_1",
            "P25_S0": "FRONTDROTR_SW_0",
            "P41_S0": "FRTDROTRLR_DOOR_DL3_0",
            "P40_S0": "FRTDROTRLR_DOOR_SG2_0",
            "P31_S0": "ROOF_SGLCAP_TK_0",
            "P28_S0": "ROOF_SW_0",
            "P11_S0": "ROOF_TK_0",
            "P11_S1": "ROOF_TK_1",
            "P11_S4": "ROOF_TK_4",
            "P16_S0": "RRDROTRLR_DOOR_DL3_0",
            "P17_S0": "RRDROTRLR_DOOR_SG2_0",
            "P29_S0": "SIDEOTRLR_SW_0",
            "P27_S0": "SIDEOTRLH_DOORLESS_SW_0",
            "P26_S0": "SIDEOTRLH_DOOR_SW_0",
            "P26_S1": "SIDEOTRLH_DOOR_SW_1",
            "P21_S0": "SIDEOTRLH_MQ4_0",
            "P2_S0": "SIDEOTRRH_DOORLESS_SW_0",
            "P3_S0": "SIDEOTRRH_DOOR_SW_0",
            "P3_S1": "SIDEOTRRH_DOOR_SW_1",
            "P1_S0": "SIDEOTRRH_MQ4_0",
            "P1_S2": "SIDEOTRRH_MQ4_2",         
            "P12_S0": "TGATE_INR_SW_0",
            "P12_S1": "TGATE_INR_SW_1",
        }
        self.panel_name_to_id = {v: k for k, v in self.panel_id_to_name.items()}
        self.panel_id_to_subgroup = {
            "SIDEOTR": [
                "P1_S0", "P1_S2", "P21_S0", "P26_S0", "P26_S1", "P3_S0", "P3_S1", "P27_S0", "P2_S0"
                ],
            "ROOF": [
                "P11_S0", "P11_S1", "P11_S4", "P28_S0", "P31_S0"
                ],
            "FENDER": [
                "P7_S0", "P7_S1", "P22_S0", "P22_S17", "P22_S32", "P23_S0", "P8_S0", "P6_S0"
                ],
            "DOOR": [
                "P25_S0", "P41_S0", "P16_S0", "P40_S0", "P17_S0", "P29_S0"
                ],
            "TGATE": [
                "P12_S0", "P12_S1"
                ]
        }
        # self.panel_name_to_subgroup = {subgroup: [self.get_panel_name(panel_id) for panel_id in panel_ids] for subgroup, panel_ids in self.panel_id_to_subgroup.items()} 
  
    @staticmethod
    def get_id_to_name_mapping():
        return MAP().panel_id_to_name

    @staticmethod
    def get_name_to_id_mapping():
        return MAP().panel_name_to_id

    @staticmethod
    def get_panel_id(panel_name):
        return MAP().panel_name_to_id.get(panel_name, None)

    @staticmethod
    def get_panel_name(panel_id):
        return MAP().panel_id_to_name.get(panel_id, None)

    @staticmethod
    def get_panel_subgroup_by_name(panel_name):
        for subgroup, panel_name_list in MAP().panel_name_to_subgroup.items():
            if panel_name in panel_name_list:
                return subgroup
        return None
    
    @staticmethod
    def get_panel_subgroup_by_id(panel_id):
        for subgroup, panel_id_list in MAP().panel_id_to_subgroup.items():
            if panel_id in panel_id_list:
                return subgroup
        return None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MAP, cls).__new__(cls)
        return cls.instance
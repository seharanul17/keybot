

def get_model(save_manager):
    from model.iterativeRefinementModels.RITM_SE_HRNet32 import RITM as Model
    model = Model(save_manager.config)
    return model
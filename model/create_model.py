from .SRResNet import create_net as load_SRResNet
from .SRResNet import SRResNet
from .FSRCNN import create_net as load_FSRCNN
from .FSRCNN import FSRCNN
from .ipt import create_net as load_ipt
from .SMFANet import create_net as load_SMFANet
from .SMFANet import SMFANet
from .MAE_MAN import create_net as load_MAE_MAN
from .WTCA import create_net as load_WTCA
from .AWAM import create_net as load_AWAM
from .AFNONET import AFNONet
from .WTCA import WTN
from .SCNet import SCNet
def load_model(model_name):
    if model_name == 'SRResNet':
        model = load_SRResNet()
        return model
    if model_name == 'SRResNet_land':
        model = SRResNet(num_channels=3)
        return model
    if model_name == 'FSRCNN':
        model = load_FSRCNN()
        return model
    if model_name == 'FSRCNN_land':
        model = FSRCNN(num_channels=3)
        return model
    if model_name == 'ipt':
        model = load_ipt()
        return model
    if model_name == 'SMFANet':
        model = load_SMFANet() 
        return model
    if model_name == 'SMFANet_land':
        model = SMFANet(in_channel=3)
        return model
    if model_name == 'MAE_MAN':
        model = load_MAE_MAN()
        return model
    if model_name =='WTN_land':
        model = WTN(n_colors=3,n_feats=128)
        return model
    if model_name == 'WTCA':
        model = load_WTCA()
        return model
    if model_name == 'AWAM':
        model = load_AWAM()
        return model
    if model_name == 'AFWT':
        model = AFNONet(img_size=(180,360), patch_size=(6,6), in_chans=6, out_chans=6)
        SR_model = WTN(n_colors=6,n_feats=128)
        return model,SR_model
    if model_name == 'AFWT_land':
        model = AFNONet(img_size=(450,900), patch_size=(6,6), in_chans=3, out_chans=3)
        SR_model = WTN(n_colors=3,n_feats=128)
        return model,SR_model
    if model_name == 'AFSMFA':
        model = AFNONet(img_size=(180,360), patch_size=(6,6), in_chans=6, out_chans=6)
        SR_model = SMFANet(in_channel=6)
        return model,SR_model
    if model_name == 'AFFC':
        model = AFNONet(img_size=(180,360), patch_size=(6,6), in_chans=6, out_chans=6)
        SR_model = FSRCNN(num_channels=6)
        return model,SR_model
    if model_name == 'AFSC':
        model = AFNONet(img_size=(180,360), patch_size=(6,6), in_chans=6, out_chans=6)
        SR_model = SCNet()
        return model,SR_model
    if model_name == 'AFSR':
        model = AFNONet(img_size=(180,360), patch_size=(6,6), in_chans=6, out_chans=6)
        SR_model = SRResNet(num_channels=6)
        return model,SR_model
        


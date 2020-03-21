import torch
from torch import nn
import sys, getopt

def main(argv):
    opts, args = getopt.getopt(argv,"hi:o:n:",["ifile=","ofile=","stepn="])
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            model_path = arg
        elif opt in ("-o", "--ofile"):
            trimmed_model_path = arg
        elif opt in ("-n", "--stepn"):
            step_n = arg
    base_size = 270
    step_size = 160
    state_dict = torch.load(model_path, map_location="cpu")
    model = state_dict['model']
    cls_weight_new = torch.Tensor((base_size+1)+step_size*step_n, 1024)
    nn.init.normal_(cls_weight_new, std=0.01)
    cls_weight_new[:(base_size+1)+step_size*(step_n-1)] = model['module.roi_heads.box.predictor.cls_score.weight']
    model['module.roi_heads.box.predictor.cls_score.weight'] = cls_weight_new
    
    reg_weight_new = torch.Tensor(((base_size+1)+step_size*step_n)*4, 1024)
    nn.init.normal_(reg_weight_new, std=0.001)
    reg_weight_new[:((base_size+1)+step_size*(step_n-1))*4] = model['module.roi_heads.box.predictor.bbox_pred.weight']
    model['module.roi_heads.box.predictor.bbox_pred.weight'] = reg_weight_new
    
    reg_bias_new = torch.Tensor(((base_size+1)+step_size*step_n)*4)
    nn.init.constant_(reg_bias_new, 0)
    reg_bias_new[:((base_size+1)+step_size*(step_n-1))*4] = model['module.roi_heads.box.predictor.bbox_pred.bias']
    model['module.roi_heads.box.predictor.bbox_pred.bias'] = reg_bias_new

    print("Deleting optimizer, scheduler, and iteration entries")
    del state_dict['optimizer']
    del state_dict['scheduler']
    del state_dict['iteration']
    
    torch.save(state_dict, trimmed_model_path)
    print(f'saved to: {trimmed_model_path}')

if __name__ == "__main__":
   main(sys.argv[1:])
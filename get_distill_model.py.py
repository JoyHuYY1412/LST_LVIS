import torch
from torch import nn
import sys, getopt

def main(argv):
    opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            model_path = arg
        elif opt in ("-o", "--ofile"):
            trimmed_model_path = arg

    state_dict = torch.load(model_path, map_location="cpu")
    print("Deleting optimizer, scheduler, and iteration entries")
    del state_dict['optimizer']
    del state_dict['scheduler']
    del state_dict['iteration']
    
    torch.save(state_dict, trimmed_model_path)
    print(f'saved to: {trimmed_model_path}')

if __name__ == "__main__":
   main(sys.argv[1:])
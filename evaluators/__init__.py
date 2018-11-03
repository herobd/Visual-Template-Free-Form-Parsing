from model.unet import UNet
from model.unet_dilated import UNetDilated
from model.sol_eol_finder import SOL_EOL_Finder
from model.detector import Detector
from model.line_follower import LineFollower

from evaluators.formsdetect_printer import FormsDetect_printer
from evaluators.formsboxdetect_printer import FormsBoxDetect_printer
from evaluators.formsboxpair_printer import FormsBoxPair_printer
from evaluators.formslf_printer import FormsLF_printer
#from evaluators.formspair_printer import FormsPair_printer
from evaluators.ai2d_printer import AI2D_printer


#def FormsPair_printer(config,instance, model, gpu, metrics, outDir=None, startIndex=None):
#    return AI2D_printer(config,instance, model, gpu, metrics, outDir, startIndex)

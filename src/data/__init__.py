import data.postprocess as dpp

from data.loaders import load_data



POSTPROCESS = {"close": dpp.closing, "close_and_fill": dpp.close_and_fill,
               "fill_holes": dpp.fill_holes}

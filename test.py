import unittest
import onnx
from utility import fetch_model

OPENPILOT_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.4/selfdrive/modeld/models/supercombo.onnx"

class TestLoad(unittest.TestCase):
    def test_load_openpilot(self):
      result = onnx.load(fetch_model(OPENPILOT_MODEL))
      import pprint
      pprint.pprint(result)

if __name__ == "__main__":
    unittest.main()

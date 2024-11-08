from nn.runner import NNRunner
from util.device import check_mps_devices

check_mps_devices()

runner = NNRunner()
runner.train()

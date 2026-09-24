from vhmodels.utils.device import resolve_torch_device


class _FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _FakeTorch:
    def __init__(self, cuda_available):
        self.cuda = _FakeCuda(cuda_available)

    @staticmethod
    def device(value):
        return f"device:{value}"


def test_auto_selects_cuda_when_available():
    assert resolve_torch_device(_FakeTorch(True)) == "device:cuda:0"


def test_auto_selects_cpu_without_cuda():
    assert resolve_torch_device(_FakeTorch(False)) == "device:cpu"


def test_explicit_device_is_preserved():
    torch = _FakeTorch(True)

    assert resolve_torch_device(torch, "cpu") == "device:cpu"
    assert resolve_torch_device(torch, "cuda:1") == "device:cuda:1"

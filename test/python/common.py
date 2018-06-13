import platform


__all__ = ['LIBNGRAPH_DEVICE', 'NgraphTest']


_ext = 'dylib' if platform.system() == 'Darwin' else 'so'

LIBNGRAPH_DEVICE = 'libngraph_device.' + _ext


class NgraphTest(object):
    test_device = "/device:NGRAPH:0"
    soft_placement = False
    log_placement = True

import platform


__all__ = ['LIBNGRAPH_DEVICE']


_ext = 'dylib' if platform.system() == 'Darwin' else 'so'

LIBNGRAPH_DEVICE = 'libngraph_device.' + _ext

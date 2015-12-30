import sys
try:
    import annfab
except ImportError:
    sys.path.append('..')
    import annfab
sys.path.append('mnist_utils')

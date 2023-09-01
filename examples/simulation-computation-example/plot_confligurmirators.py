import pycwtmm.plotting as cwtpl
import pycwtmm.modulus_maxima as cmm
import matplotlib.pyplot as plt
import pickle
import sys

def main():
    plt.style.use('style.mplstyle')

    files = sys.argv[1:]
    for fn in files:
        print('start', fn)
        fig = plt.figure(figsize=(12, 8), layout='constrained')
        dat = load_wtmmizer(fn)
        cwtpl.plot_summary(dat, fig)
        fig.savefig(f'{fn.removesuffix(".xz")}.png', dpi=300)
        print('done', fn)

RetType = cmm.Wtmmizer
def load_wtmmizer(file: str) -> RetType:
    with open(file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    main()

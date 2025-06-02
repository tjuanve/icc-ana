import matplotlib as mpl

rcParams = {
    # "figure.figsize": [10.0, 7.0],
    "figure.titlesize": 'large',  # or x-large, xx-large, larger
    "axes.titlesize": 'large',
    "axes.labelsize": 'large',
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
    "legend.fontsize": 'large',
    "axes.grid": True,
    "lines.linewidth": 2.0,
    "patch.linewidth": 2.0,
    "grid.linestyle": ":",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}

mpl.rcParams.update(rcParams)

# to use latex to render fonts
if False:
    with_latex = {"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex or pdflatex
                  "text.usetex": True,                # use LaTeX to write all text
                  "text.latex.unicode": True,
                  "font.family": "serif",
                  "font.sans-serif": "cm",
                  "mathtext.fontset": "cm",
                  #"pgf.preamble": [r"\usepackage{graphicx}"]
                  "axes.titlesize"  : 20,
                  "axes.labelsize"  : 20,
                  "xtick.labelsize" : 20,
                  "ytick.labelsize" : 20,
                  "legend.fontsize" : 20,
                  }

    mpl.rcParams.update(with_latex)

# to use latex to render the whole plot (pgf)
if False:
    with_latex = {"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex or pdflatex
                  "text.usetex": True,                # use LaTeX to write all text
                  "pgf.rcfonts": False,               # don't setup fonts from rc parameters
                  "text.latex.unicode": True,
                  "font.family": "serif",
                  "font.sans-serif": "cm",
                  "mathtext.fontset": "cm",
                  "pgf.preamble": [r"\usepackage[separate-uncertainty=true]{siunitx}"],
                  "text.latex.preamble": [r"\usepackage[separate-uncertainty=true]{siunitx}"],
                  "figure.figsize"  : [5.97508875, 4.18256213],
                  "axes.titlesize"  : 11,
                  "axes.labelsize"  : 11,
                  "xtick.labelsize" : 11,
                  "ytick.labelsize" : 11,
                  "legend.fontsize" : 11,
                  "grid.linestyle"  : ":",
                  }

    mpl.rcParams.update(with_latex)

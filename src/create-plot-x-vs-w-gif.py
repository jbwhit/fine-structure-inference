
sub_264 = df_a[df_a.system==264][['wavelength', 'x', 'vshift', 'sigma']].copy()

sub_264['x_1000'] = sub_264.x + 6000.0

!rm ../img/*transition.png

calc_array = np.linspace(0.0, 2.0 * np.pi, 150)

def fraction(index):
    return (np.cos(index) + 1.0) / 2.0

xlimmin = []
xlimmax = []
for index, infrac in enumerate(calc_array):
    frac = fraction(infrac)
    xlimmax.append(np.max(sub_264.wavelength * frac + sub_264['x_1000'] * (1.0 - frac)) + 50.0)
    xlimmin.append(np.min(sub_264.wavelength * frac + sub_264['x_1000'] * (1.0 - frac)) - 50.0)

for index, infrac in enumerate(calc_array):
    frac = fraction(infrac)
    with sns.axes_style(rc={'axes.grid':False}):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(sub_264.wavelength * frac + sub_264['x_1000'] * (1.0 - frac), sub_264.vshift)
        ax.errorbar(sub_264.wavelength * frac + sub_264['x_1000'] * (1.0 - frac), sub_264.vshift,
                    yerr=sub_264.sigma,  ls='none',
                    color=sns.color_palette()[0])
        ax.hlines(0, 
                  np.min(sub_264.wavelength * frac + sub_264['x_1000'] * (1.0 - frac)),
                  np.max(sub_264.wavelength * frac + sub_264['x_1000'] * (1.0 - frac)),
                  linestyles=':',
                  linewidth=0.5,
                  color='k'
                 )
        ax.set_xlim(xlimmin[index], xlimmax[index])
        ax.set_ylabel("vshift [m/s]")
        ax.set_xticks([])
        if frac < 0.05:
            ax.set_xlabel("X")
        elif frac > 0.95:
            ax.set_xlabel("wavelength")
        else:
            ax.set_xlabel(" switching-axis ")
        fig.tight_layout()
        fig.savefig("../img/" + str(round(index, 2) + 1000) + ".transition.png")
        plt.close()
    
!convert -delay 5 -loop 0 ../img/*transition.png ../img/01-animation.gif

!convert ../img/01-animation.gif -fuzz 10% -layers Optimize ../img/02-animation.gif
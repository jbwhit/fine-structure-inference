
def extra_eda(row):
    """King 2012 has extra 'random' systematic error added in quadrature."""
    if row.sigflag == 3:
        extra = 9.05
    elif row.sigflag == 2:
        extra = 17.43
    else:
        extra = 0.0
    return np.sqrt(row.eda ** 2.0 + extra ** 2.0)

def assign_dipole(row):
    """Assign best-fit dipole from King 2012 to column in dataset."""
    return dipole_alpha(row['#J2000'])

def assign_dipole_angle(row):
    """King 2012 angle from pole to position on sky via J2000."""
    return j2000_to_theta(row['#J2000'])

def parse_j2000(name):
    """Takes the J2000 name stored in the results and returns it in a format astropy can understand."""
    return ' '.join([name[1:3], name[3:5], name[5:7], name[7:10], name[10:12], name[12:]])

def j2000_to_theta(name):
    """Returns the angle (degrees) between the position on the sky from 
    a given `name` and the position of the dipole model from 2012, King."""
    c = SkyCoord(parse_j2000(name), unit=(u.hourangle, u.deg))
    return float(c.separation(dipole).to_string(decimal=True))

def dipole_alpha(name):
    """Returns the value of Delta alpha/alpha as given by the best fit 2012 King model for 
    the given name (position).
    """
    theta = j2000_to_theta(name)
    return (DIP_AMPLITUDE * np.cos(np.deg2rad(theta)) + DIP_MONOPOLE) * 1e6


full_parse = pd.read_csv("../data/full-parse.tsv", sep='\t')
full_parse['extraeda'] = full_parse.apply(extra_eda, axis=1)
full_parse['dipole_fit'] = full_parse.apply(assign_dipole, axis=1)
full_parse['dipole_angle'] = full_parse.apply(assign_dipole_angle, axis=1)

full_parse = full_parse.rename(columns={"#J2000":"J2000",
'zem': 'z_emission',
'zabs': 'z_absorption',
'da': 'delta_alpha',
'eda': 'error_delta_alpha',
'extraeda': 'extra_error_delta_alpha',
'dipole_fit': 'dipole_delta_alpha',
'dipole_angle': 'dipole_angle',
'sample': 'sample',
'source': 'source',
'sigflag': 'sigflag',
'imrotator': 'imrotator',
'transition': 'transitions',
})

full_parse[['J2000',
'z_emission',
'z_absorption',
'delta_alpha',
'error_delta_alpha',
'extra_error_delta_alpha',
'dipole_delta_alpha',
'dipole_angle',
'sample',
'source',
'sigflag',
'imrotator',
'transitions',
]].to_csv("../data/full-parse-new.tsv", sep='\t', index=False)
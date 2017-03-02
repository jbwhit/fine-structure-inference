sheets = ['B346', 'B390', 'B437', 'R564', 'R760', 'R860',]
# https://docs.google.com/spreadsheets/d/1lGW4GVVKO6mTxkGR4v5sc9P9-ITFwAf2vf6yn-sdQGE/edit?usp=sharing
ccd = pd.read_excel("/Users/jonathan/junk/uves-vlt-ccd.xlsx", sheetname=sheets)
ccd.keys()

columns = ['chip', 'wav_min', 'midpoint', 'wav_max']
chips = pd.DataFrame(columns=columns)
for index, sheet in enumerate(sheets):
    wav_min = ccd[sheet]['wav_start_nm'].min() * 10.0
    wav_max = ccd[sheet]['wav_end_nm'].max() * 10.0
    midpoint = np.average([wav_min, wav_max])
    chips.loc[index] = np.array([sheet, wav_min, midpoint, wav_max])

chips = chips.set_index('chip')

chips.to_csv("../data/vlt-ccd.csv")
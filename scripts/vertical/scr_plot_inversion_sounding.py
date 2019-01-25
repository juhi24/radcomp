# coding: utf-8

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from radcomp.vertical import plotting, m2km

plt.ion()
plt.close('all')

tpath='/home/jussitii/results/radcomp/vertical/soundings/140222_00.txt'
skiprows = (0,1,2,4,5)
data = pd.read_table(tpath, delim_whitespace=True, index_col=0, skiprows=skiprows).dropna()

fig = plt.figure(figsize=(2.5,3), dpi=140)
plt.plot(data.TEMP, data.HGHT)
ax = plt.gca()
#ax.invert_yaxis()
ax.set_xlim(left=-35)
#ax.set_ylim(top=500)
ax.set_ylim(bottom=0, top=10000)
ax.axvline(-10, linestyle='--', color='black')
ax.axvline(-20, linestyle='--', color='black')
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(m2km))
ax.set_xticks((-30, -20,-10,0))
ax.set_xlabel(plotting.LABELS['temp_mean'])
#ax.set_ylabel('$p$, hPa')
ax.set_ylabel('Height, km')
ax.set_title('Sounding, 1 Feb 2014')
fig.set_tight_layout(True)
fig.savefig('/home/jussitii/results/radcomp/vertical/soundings/inversion_sample.png')

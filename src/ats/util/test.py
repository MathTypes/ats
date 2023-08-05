import plotly
plotly.__version__ #5.5.0

import kaleido #required
kaleido.__version__ #0.2.1

#now this works:
import plotly.graph_objects as go

fig = go.Figure()
fig.write_image('aaa.png')


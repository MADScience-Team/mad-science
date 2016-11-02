from operator import itemgetter
import itertools, os
import logging

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool, BoxZoomTool, WheelZoomTool, ResetTool, PanTool, BoxSelectTool, TapTool
from bokeh.models import ColorBar, LinearColorMapper, FixedTicker, LabelSet, OpenURL
import bokeh.palettes
from bokeh.models.widgets import Div, PreText,  DataTable, TableColumn, Slider, Select
from bokeh.layouts import layout, widgetbox
from bokeh.io import curdoc

import pandas as pd
import numpy as np
from tabulate import tabulate
import pickle
import bhtsne



## Use scikit-learn for K-Means clustering: identify topics.
from sklearn.cluster import KMeans

data_directory = 'data/'
model_directory = 'models/'

from imp import reload
reload(logging)

LOG_FILENAME = data_directory + 'vec2topic.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s',"%b-%d-%Y %H:%M:%S")
logger.handlers[0].setFormatter(formatter)


## Load model: words, vectors, and their computed metrics.
with open('models/word_d2v.pkl','r') as fp:
    word_d2v = pickle.load(fp)
        
with open('models/data_d2v.pkl','r') as fp:
    data_d2v = pickle.load(fp)

with open('models/metric.pkl','r') as fp:
    metric = pickle.load(fp)

## Compute 2D embedding of word vectors.
if not os.path.isfile('models/X_2D.pkl'):
  X_2D = bhtsne.tsne(np.array(data_d2v))
  with open('models/X_2D.pkl','w') as fp:
      pickle.dump(X_2D,fp)
else:
  with open('models/X_2D.pkl','r') as fp:
    X_2D = pickle.load(fp)
  logger.info('Read X_2D from disk: %s', X_2D.shape)

##Kmeans
NUM_TOPICS = 50
K=NUM_TOPICS
kmeans=KMeans(n_clusters=K)
kmeans.fit([w for w in data_d2v])
kmeans_label={word_d2v[x]:kmeans.labels_[x] for x in xrange(len(word_d2v))}

kmeans_label_ranked={}

#metrics_clean = {}
#for w in word_d2v:
#  if w in metric:
#    metrics_clean[w] = metric[w]
#
#metric = metrics_clean

topic=[[] for i in xrange(K)]
clust_depth=[[] for i in xrange(K)]
print type(metric)
for i in xrange(K):
    topic[i]=[word_d2v[x] for x in xrange(len(word_d2v)) if kmeans.labels_[x]==i]
    temp_score=[metric[w] for w in topic[i]]
    clust_depth[i]=-np.mean(sorted(temp_score,reverse=True)[:])#]int(np.sqrt(len(topic[i])))])

index=np.argsort(clust_depth)
for num,i in enumerate(xrange(K)):
  for w in topic[index[i]]:
    kmeans_label_ranked[w]=i

print np.mean(sorted(temp_score,reverse=True)[:]), index
for w in topic[index[8]]: print w,
print
for w in [w[0] for w in sorted([[w,metric[w]] for w in topic[index[8]]],key=itemgetter(1),reverse=True)]:
    print w,

lister=[]
to_show=K
to_show_words=200 #the maximum number of words of each type to display
for i in xrange(to_show):
  top=topic[index[i]]
  sort_top=[w[0] for w in sorted([[w,metric[w]] for w in top],key=itemgetter(1),reverse=True)]
  lister.append(['Topic %d' %(i+1)]+sort_top[:to_show_words])

max_len=max([len(w) for w in lister])
new_list=[]
for list_el in lister:
  new_list.append(list_el + [''] * (max_len - len(list_el)))

Topics=list(itertools.izip_longest(*new_list))
#X.insert(len(X),[-int(clust_depth[index[w]]*100)*1./100 for w in xrange(K)])
score_words=[w[0] for w in sorted(metric.items(),key=itemgetter(1),reverse=True)][:to_show_words]

df_tmp = pd.DataFrame(new_list).T
df_new = pd.DataFrame(df_tmp[1:len(new_list)].values,columns=[l[0] for l in new_list])
df_new = pd.DataFrame(df_new['Topic 8'])

df = pd.DataFrame(zip(word_d2v,[kmeans_label[w] for w in word_d2v],
                      [metric[w] for w in word_d2v]),columns=['word','topic','metric'])

num_top_scoring_words = 500
kmeans_label_color = [kmeans_label_ranked[w]+1 for w in word_d2v]
top_scoring_words = [(w,d[0],d[1],k,m) for w,d,k,m in zip(word_d2v,X_2D,kmeans_label_color,[metric[ww] for ww in word_d2v]) 
                           if w in score_words]
#top_df = pd.DataFrame(top_scoring_words, columns=['word','x','y','topic','metric'])
#top_df = top_df[(top_df.topic==8) | (top_df.topic==2)].sort_values('metric', ascending=False)

#topic_number = Slider(title="Topic Number", start=1, end=K, value=11, step=1)
topic_number = Select(title="Topic Number", value='Topic 1', options=['Topic ' + str(j) for j in range(1,K+1)])
source_topic = ColumnDataSource(data=dict(x=[], y=[], names=[]))

def select():
  topic_no = topic_number.value
  topic_no = int(topic_no.split(' ')[1])
  top_df = pd.DataFrame(top_scoring_words, columns=['word','x','y','topic','metric'])
  top_df = top_df[(top_df.topic==topic_no)].sort_values('metric', ascending=False)
  return top_df

def update():
  df = select()
  source_topic.data = dict(
          x=df.x,
          y=df.y,
          names=list(df.word),
        )

  source_topics.data = dict(
          x=df.x,
          y=df.y,
          word=df.word,
          topic=df.topic,
          metric=df.metric,
        )


topic_number.on_change('value', lambda attr, old, new: update())

#kmeans_colors = bokeh.palettes.plasma(50) #bokeh.palettes.brewer['RdYlGn'][10]
kmeans_colors = bokeh.palettes.brewer['RdYlGn'][10] + bokeh.palettes.plasma(50)
colors = [kmeans_colors[kmeans_label_color[x]-1] for x in range(len(word_d2v))]
x=X_2D[:,0]
y=X_2D[:,1]
source = ColumnDataSource( data=dict(
  x=x,
  y=y,
  word=word_d2v,
  topic = kmeans_label_color,
  metric = df['metric'],
 )
)

#top_scores = ColumnDataSource(
#  data=dict(
#    x=top_df.x,
#    y=top_df.y,
#    names=list(top_df.word)
#  )
#)

labels = LabelSet(x='x', y='y', text='names', level='glyph',
              x_offset=5, y_offset=5, source=source_topic, render_mode='canvas', 
              border_line_color='black', border_line_alpha=1.0,
              background_fill_color='white', background_fill_alpha=1.0)

radii = [metric[w] for w in word_d2v]

source_topics = ColumnDataSource(data=dict(x=[],y=[],word=[],metric=[]))
columns = [
             TableColumn(field="word", title="words"),
	     TableColumn(field="topic", title="topics"),
	     TableColumn(field="metric", title="metrics"),
	  ]

#data_columns = [TableColumn(field=df_new.columns[i],title=df_new.columns[i]) for i in range(K)]
#data_table = DataTable(source=source_topics, columns=data_columns, width=400, height=280)

data_columns = [TableColumn(field='Topic 8',title='Topic 8')]
data_table = DataTable(source=source_topics, columns=columns, width=400, height=500)
table = widgetbox(topic_number,data_table)

color_mapper = LinearColorMapper(kmeans_colors,low=0,high=9)

p = figure(plot_width=800, plot_height=600, title="Distributed Word Embeddings: " + str(K) + " Topics", 
           tools=['box_zoom', 'box_select', 'pan','wheel_zoom','reset', 'save','tap'])

cr = p.circle('x', 'y', source=source, radius=radii, color=colors, fill_alpha=0.4, line_color=None, hover_line_color="white",)
hover = HoverTool( tooltips=[ #("index", "$index"), 
                              #("(x,y)", "($x, $y)"), 
			      ("word", "@word"),
			      ("topic","@topic"),
			      ("metric", "@metric"), ], renderers=[cr], mode='mouse',) 
p.add_tools(hover)
p.add_layout(labels)
p.xaxis.visible=False
p.yaxis.visible=False
p.grid.visible=False
color_bar = ColorBar(color_mapper=color_mapper, orientation='horizontal',
                     location='bottom_left', scale_alpha=0.7)
		     #ticker=FixedTicker(ticks=[2,6,10,14,18]))
		     #p.add_layout(color_bar) 
topic_legend = "topics"
#p.circle('x', 'y', source=source, size=10, color=colors, legend=topic_legend)

first=1;last=20
html = tabulate([Topics[i][first-1:last] for i in range(0,21)], tablefmt=u'psql')
div = widgetbox(PreText(text=html,width=2000))

url = 'https://www.google.com/search?q=@word'
taptool = p.select(type=TapTool)
taptool.callback = OpenURL(url=url)

#grid = gridplot([[p, table],[div]])
#grid = gridplot([[p],[div]])
grid = layout([[p, table], [div]])

from bokeh.embed import components
from bokeh.resources import CDN
# Generate the script and HTML for the plot
script, div = components(grid)

# Return the webpage
html = """
<!doctype html>
<head>
 <title></title>
  {bokeh_css}
</head>
<body>
 {div}
 {bokeh_js}
 {script}
</body>
""".format(script=script, div=div, bokeh_css=CDN.render_css(), bokeh_js=CDN.render_js())

curdoc().add_root(grid)
curdoc().title = "Distributed Word Embeddings: " + str(K) + " Topics"

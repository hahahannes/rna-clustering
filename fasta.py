from Bio import SeqIO
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data():
   # https://lncipedia.org/download
   data_dict = {'sequence': [], 'length': [], 'ratio_g': [], 'ratio_t': [], 'ratio_c': [], 'ratio_a': []}
   data = SeqIO.parse("data/lncipedia_5_2.fasta", "fasta")
   for i, record in enumerate(data):
      #print("Id: %s" % record.id) 
      #print("Name: %s" % record.name) 
      #print("Description: %s" % record.description) 
      #print("Annotations: %s" % record.annotations) 
      #print("Sequence Alphabet: %s" % record.seq.alphabet)record.seq
      data_dict['sequence'].append(record.seq)
      length = len(record.seq)
      data_dict['length'].append(length)

      count_g = 0
      count_a = 0
      count_t = 0
      count_c = 0

      for c in record.seq:
         if c == 'G':
            count_g += 1
         elif c == 'T':
            count_t += 1
         elif c == 'C':
            count_c += 1
         elif c == 'A':
            count_a += 1

      data_dict['ratio_g'].append(count_g/length*100)
      data_dict['ratio_t'].append(count_t/length*100)
      data_dict['ratio_c'].append(count_c/length*100)
      data_dict['ratio_a'].append(count_a/length*100)

      if i == 2000:
         break
   df = pd.DataFrame.from_dict(data_dict)
   # df = StandardScaler().fit_transform(X)
   return df

def dbscan(df, features):
   df = df[features]
   db = DBSCAN(eps=0.7, min_samples=5).fit(df)
   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   core_samples_mask[db.core_sample_indices_] = True
   labels = db.labels_

   # Number of clusters in labels, ignoring noise if present.
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise_ = list(labels).count(-1)

   print('Estimated number of clusters: %d' % n_clusters_)
   print('Estimated number of noise points: %d' % n_noise_)

   # Black removed and is used for noise instead.
   unique_labels = set(labels)
   colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
   for k, col in zip(unique_labels, colors):
      if k == -1:
         # Black used for noise.
         col = [0, 0, 0, 1]

      class_member_mask = (labels == k)

      xy = df[class_member_mask & core_samples_mask]
      plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=14)

      xy = df[class_member_mask & ~core_samples_mask]
      plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=6)

   plt.title('Estimated number of clusters: %d' % n_clusters_)

def kmeans(df, features, n_clusters):
   df = df[features]
   kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=3)
   kmeans.fit(df)
   LABEL_COLOR_MAP = {0 : 'r',
                      1 : 'k',
                      2: 'b',
                      3: 'g'}
   label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
   centroids = kmeans.cluster_centers_
   if len(features) > 1:
      plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='g', zorder=10)
      plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=label_color)
   else:
      plt.scatter(centroids[:, 0], np.zeros(shape=(n_clusters,1)), marker='x', color='g', zorder=10)
      hist, edges = np.histogram(df.iloc[:, 0], bins=100)
      def add_counts(f):
         found_edge = 0
         found_count = 0
         for histogram in zip(edges, hist):
            edge = histogram[0]
            count = histogram[1]
            if edge > f[0]:
               break
            else:
               found_edge = edge
               found_count = count
         f['count'] = found_count
         return f
      df = df.apply(axis=1, func=add_counts)
      plt.scatter(df['length'], df['count'], c=label_color)

      df = df.sort_values(by='length', axis=0)
      plt.plot(df['length'], df['count'])

# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/scatter_hist.html

def pair(df):
   cols = ['length', 'ratio_g', 'ratio_a', 'ratio_c', 'ratio_t']
   pp = sns.pairplot(df[cols], size=1.8, aspect=1.8,
                     plot_kws=dict(edgecolor="k", linewidth=0.5),
                     diag_kind="kde", diag_kws=dict(shade=True))

   fig = pp.fig 
   fig.subplots_adjust(top=0.93, wspace=0.3)

df = load_data()
kmeans(df, ['length'], 3)
plt.show()
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
from pybedtools import BedTool
from scipy.spatial.distance import cdist 
from BCBio.GFF import GFFExaminer
from BCBio import GFF
from statistics import mean

def calc_number_introns(start_pos, end_pos, list_of_exons_coordinates):
   seq_starting_with_exon = False
   seq_ending_with_exon = False
   num_exons = len(list_of_exons_coordinates)
   for exon in list_of_exons_coordinates:
      exon_start_pos = exon[0]
      exon_end_pos = exon[1]
      if exon_start_pos == start_pos:
         seq_starting_with_exon = True
      if exon_end_pos == end_pos:
         seq_ending_with_exon = True 

   if seq_starting_with_exon and seq_ending_with_exon:
      return num_exons - 1
   elif (seq_starting_with_exon and not seq_ending_with_exon) or (seq_ending_with_exon and not seq_starting_with_exon):
      return num_exons
   else:
      return num_exons + 1

def calc_mean_exon_length(list_of_exons_coordinates):
   lengths = [exon[1] - exon[0] for exon in list_of_exons_coordinates]
   return mean(lengths)


def load_data(n_row=None):
   # https://lncipedia.org/download
   data_dict = {'length': [], 'ratio_g': [], 'ratio_t': [], 'ratio_c': [], 'ratio_a': [], 'number_exons': [], 'chromosom': [], 'start_pos': [], 'end_pos': [], 'length_from_pos': [], 'number_introns': [], 'mean_exon_length': []}
   fasta_data = SeqIO.parse("data/lncipedia_5_2.fasta", "fasta")
   bed_raw_data = BedTool('data/lncipedia.bed')
   examiner = GFFExaminer()
   in_handle = open('data/lncipedia_5_2_hg38.gff')
   annotation_data = {}
   for i, rec in enumerate(GFF.parse(in_handle)):
      # chromosom e.g. chr1
      for feature in rec.features:
         # lncRNA eg. LNC1725
         if not feature.type == 'lnc_RNA':
            break

         exon_locations = []
         lnc_id = feature.id
         for sub_feature in feature.sub_features:
            if sub_feature.type == 'exon':
               exon = (sub_feature.location.start, sub_feature.location.end)
               exon_locations.append(exon)

         annotation_data[lnc_id] = exon_locations
      
   in_handle.close()
   bed_data = {}
   
   for record in bed_raw_data:
      bed_data[record.name] = {
         'number_exons': int(record.fields[9]),
         'chromosom': record.fields[0],
         'start_pos': int(record.fields[1]), # im bed -1 im vgl zu gff und online
         'end_pos': int(record.fields[2])
      }

   for i, record in enumerate(fasta_data):
      length = len(record.seq)
      data_dict['length'].append(length)
      for bed_feature in ['number_exons', 'chromosom', 'start_pos', 'end_pos']:
         data_dict[bed_feature].append(bed_data[record.name][bed_feature])

      end_pos = bed_data[record.name]['end_pos']
      start_pos = bed_data[record.name]['start_pos']
      exon_locations = annotation_data[record.id]
      data_dict['length_from_pos'].append(end_pos - start_pos)
      data_dict['number_introns'].append(calc_number_introns(start_pos, end_pos, exon_locations))
      data_dict['mean_exon_length'].append(calc_mean_exon_length(exon_locations))

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

      if n_row:
         if i == n_row:
            break
   df = pd.DataFrame.from_dict(data_dict)
   df['chromosom'] = df['chromosom'].apply(lambda x: x.split('chr')[1])
   df = df[(df['chromosom'] != 'X') & (df['chromosom'] != 'Y')]
   df['chromosom'] = pd.to_numeric(df['chromosom'])
   df = df.apply(lambda x: (x-x.mean()) / x.std(), axis=0)

   return df

def fit_dbscan(df, eps=0.2, min_samples=3):
   db = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   core_samples_mask[db.core_sample_indices_] = True
   labels = db.labels_

   # Number of clusters in labels, ignoring noise if present.
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise_ = list(labels).count(-1)

   return (n_clusters_, n_noise_, labels, core_samples_mask)

def plot_dbscan(df, labels, core_samples_mask, n_clusters_, ax):
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
      ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=14)

      xy = df[class_member_mask & ~core_samples_mask]
      ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=6)

   #ax.title('Estimated number of clusters: %d' % n_clusters_)
   ax.set_xlabel(df.columns[0])
   ax.set_ylabel(df.columns[1])
   output_path = './output/dbscan' 
   #plt.savefig('%s/%s-%s.png' % (output_path, df.columns[0], df.columns[1]))
   #plt.clf()
   #plt.close()

def plot_dbscan_3d(df, labels, core_samples_mask, n_clusters_, ax):
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
      ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], xy.iloc[:, 2], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=14)

      xy = df[class_member_mask & ~core_samples_mask]
      ax.plot(xy.iloc[:, 0], xy.iloc[:, 1], xy.iloc[:, 2], 'o', markerfacecolor=tuple(col),
               markeredgecolor='k', markersize=6)

   #ax.title('Estimated number of clusters: %d' % n_clusters_)
   ax.set_xlabel(df.columns[0])
   ax.set_ylabel(df.columns[1])
   ax.set_zlabel(df.columns[2])

def plot_kmeans_2d(centroids, df, n_clusters, model, ax):
   feature1 = df.columns[0]
   feature2 = df.columns[1]
   ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='b', zorder=10)
   ax.scatter(df[feature1], df[feature2], c=model.predict(df))
   ax.set_xlabel(feature1)
   ax.set_ylabel(feature2)
   #plt.savefig('%s/%s-%s.png' % (output_path, feature1, feature2))
  

def plot_kmeans_3d(model, df, ax):
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2],
               c=model.predict(df), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    ax.dist = 12

def plot_kmeans_1d(centroids, df, n_clusters, model, ax):
   feature1 = df.columns[0]
   ax.scatter(centroids[:, 0], np.zeros(shape=(n_clusters,1)), marker='x', color='g', zorder=10)
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
   ax.scatter(df[feature1], df['count'], c=model.predict(df[[feature1]]))
   df = df.sort_values(by=feature1, axis=0)
   ax.plot(df[feature1], df['count'])
   ax.set_xlabel(feature1)
   ax.set_ylabel('Haeufigkeit')
   #plt.savefig('%s/%s.png' % (output_path, feature1))
   #plt.clf()
   #plt.close()

def run_dbscan(df, eps):
   n_clusters_, n_noise_, labels, core_samples_mask = fit_dbscan(df, eps)
   plot_dbscan(df, labels, core_samples_mask, n_clusters_)

def run_all_clustering(df_all, features):
   k_means_number_clusters = 8
   for i, feature1 in enumerate(features):
      df = df_all[[feature1]]
      run_kmeans(df, k_means_number_clusters)

      for feature2 in features[i+1:]:
         df = df_all[[feature1, feature2]]
         run_kmeans(df, [feature1, feature2], k_means_number_clusters)
         run_dbscan(df, [feature1, feature2])

def fit_kmeans(df, n_clusters):
   kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=3)
   cluster_labels = kmeans.fit_predict(df)
   return (kmeans, kmeans.cluster_centers_, cluster_labels)

def run_kmeans(df, n_clusters):
   output_path = './output/kmeans' 
   model, centroids, labels = fit_kmeans(df, n_clusters)

   if len(df.columns) == 1:
      plot_kmeans_1d(centroids, df, n_clusters, model)      
   else:
      plot_kmeans_2d(centroids, df, n_clusters, model)
      


# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/scatter_hist.html

def pair(df, features):
   pp = sns.pairplot(df[features], size=1.8, aspect=1.8,
                     plot_kws=dict(edgecolor="k", linewidth=0.5),
                     diag_kind="kde", diag_kws=dict(shade=True))

   fig = pp.fig 
   fig.subplots_adjust(top=0.93, wspace=0.3)
   pp.savefig('output/pair_plot.png')

def calc_distortion(df, cluster_centers):
   return sum(np.min(cdist(df, cluster_centers, 'euclidean'),axis=1)) / df.shape[0] 

def plot_elbow(K, distortions):
   plt.plot(K, distortions, 'bx-') 
   plt.xlabel('Anzahl Cluster k') 
   plt.ylabel('Silhouette Score') 
   plt.title('Silhouette Score je Anzahl Cluster') 
   plt.show() 

def find_best_number_clusters(df, k_range=[2,4,6,8,10,12,14,16,18,20]):
   distortions = []

   for k in k_range:
      model, cluster_centers, labels = fit_kmeans(df, k)
      #distortions.append(calc_distortion(df, cluster_centers))
      distortions.append(metrics.silhouette_score(df, labels))

   return [k_range[distortions.index(max(distortions))], distortions]

def find_best_number_clusters_sum_squares(df, k_range=[2,4,6,8,10,12,14,16,18,20]):
   distortions = []

   for k in k_range:
      model, cluster_centers, labels = fit_kmeans(df, k)
      distortions.append(model.inertia_)

   return [k_range[distortions.index(max(distortions))], distortions]

def find_best_eps(df, eps_range=[0.001, 0.1, 0.5, 1, 2, 4, 5, 10, 20]):
   silouettes = []
   for ep in eps_range:
      n_clusters_, n_noise_, labels, core_samples_mask = fit_dbscan(df, ep)
      if n_clusters_ > 1:
         silhouette_score = metrics.silhouette_score(df, labels)
         silouettes.append(silhouette_score)
   return [eps_range[silouettes.index(max(silouettes))], silouettes] if silouettes else [None, None]

def plot_best_eps(eps, silouettes):
   plt.plot(eps, silouettes, 'bx-') 
   plt.xlabel('Values of eps') 
   plt.ylabel('Silouette Score') 
   plt.title('Silhouette score per Epsilon') 
   plt.show() 

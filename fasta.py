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

def load_data():
   # https://lncipedia.org/download
   data_dict = {'length': [], 'ratio_g': [], 'ratio_t': [], 'ratio_c': [], 'ratio_a': [], 'number_exons': [], 'chromosom': [], 'start_pos': [], 'end_pos': [], 'length_from_pos': [], 'number_introns': []}
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

      if i == 10000:
         break
   df = pd.DataFrame.from_dict(data_dict)
   return df

def fit_dbscan(df, eps=0.2):
   db = DBSCAN(eps=eps, min_samples=5).fit(df)
   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   core_samples_mask[db.core_sample_indices_] = True
   labels = db.labels_

   # Number of clusters in labels, ignoring noise if present.
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
   n_noise_ = list(labels).count(-1)

   return (n_clusters_, n_noise_, labels, core_samples_mask)

def run_dbscan(df, features, eps):
   output_path = './output/dbscan' 
   feature1 = features[0]
   feature2 = features[1]
   n_clusters_, n_noise_, labels, core_samples_mask = fit_dbscan(df, eps)
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
   plt.xlabel(feature1)
   plt.ylabel(feature2)
   plt.savefig('%s/%s-%s.png' % (output_path, feature1, feature2))
   plt.clf()
   plt.close()

def run_all_clustering(df_all, features):
   k_means_number_clusters = 8
   for i, feature1 in enumerate(features):
      df = df_all[[feature1]]
      run_kmeans(df, [feature1], k_means_number_clusters)

      for feature2 in features[i+1:]:
         df = df_all[[feature1, feature2]]
         df = df.apply(lambda x: (x-x.mean()) / x.std(), axis=0)
         run_kmeans(df, [feature1, feature2], k_means_number_clusters)
         run_dbscan(df, [feature1, feature2])

def fit_kmeans(df, n_clusters):
   kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=3)
   kmeans.fit(df)
   return (kmeans, kmeans.cluster_centers_)

def run_kmeans(df, features, n_clusters):
   output_path = './output/kmeans' 
   
   if len(features) == 1:
      feature1 = features[0]
      model, centroids = fit_kmeans(df, n_clusters)
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
      plt.scatter(df[feature1], df['count'], c=label_color)
      df = df.sort_values(by=feature1, axis=0)
      plt.plot(df[feature1], df['count'])
      plt.xlabel(feature1)
      plt.ylabel('Haeufigkeit')
      plt.savefig('%s/%s.png' % (output_path, feature1))
      plt.clf()
      plt.close()
   else:
      feature1 = features[0]
      feature2 = features[1]
      model, centroids = fit_kmeans(df, n_clusters)
      plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='b', zorder=10)
      plt.scatter(df[feature1], df[feature2], c=model.predict(df))
      plt.xlabel(feature1)
      plt.ylabel(feature2)
      plt.savefig('%s/%s-%s.png' % (output_path, feature1, feature2))
      plt.clf()
      plt.close()


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
   plt.xlabel('Values of K') 
   plt.ylabel('Distortion') 
   plt.title('The Elbow Method using Distortion') 
   plt.show() 

def find_best_number_clusters(df):
   distortions = []
   K = [2,4,6,8,10,12,14,16,18,20]

   for k in K:
      label_color, cluster_centers = fit_kmeans(df, k)
      distortions.append(calc_distortion(df, cluster_centers))
   plot_elbow(K, distortions)

def find_best_eps(df):
   eps = [0.001, 0.1, 0.5, 1, 2, 4, 5]
   for ep in eps:
      n_clusters_, n_noise_, labels, core_samples_mask = fit_dbscan(df, ep)
      silhouette_score = metrics.silhouette_score(df, labels)
      print('%s: %s' % (ep, silhouette_score))

df = load_data()
features = ['length', 'ratio_g', 'ratio_a', 'ratio_c', 'ratio_t', 'number_exons', 'length_from_pos', 'number_introns']
#pair(df, features)
run_all_clustering(df, features)
#run_dbscan(df[['length', 'number_exons']], ['length', 'number_exons'], 40)
#run_kmeans(df, ['length','number_exons'], 8)

df = df[['length', 'number_exons']]
find_best_eps(df)
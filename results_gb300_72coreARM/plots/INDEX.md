# Figures in results_gb300_72coreARM

Written by `python scripts/plot/make_all.py`. Each figure is a .png in `plots/<topic>/<level>/`; data paths are relative to this results folder.

## quality / primary

| figure | what it shows | data |
|---|---|---|
| `quality/primary/quality_chembl34` | Cluster quality against chemistry (iSIM, scaffold purity, singletons, AMI and ARI with Murcko scaffolds and ChEMBL targets) of the four implementations on ChEMBL34, k = 3.5: mean and 95 % interval over input orders | data/quality/by_run, by_run_earlier_labels |

## quality / secondary

| figure | what it shows | data |
|---|---|---|
| `quality/secondary/agreement_by_library` | Agreement between runs (as agreement_chembl34) for every library, k = 3.5 | data/quality/agreement.csv |
| `quality/secondary/agreement_chembl34` | Agreement (AMI, ARI) between runs on different input orders, ChEMBL34, k = 3.5: each implementation against itself and against bblean serial, next to the floor (bblean serial against itself); all molecules and the 20 and 50 largest clusters; mean and 95 % interval over pairs of orders | data/quality/agreement.csv |
| `quality/secondary/agreement_vs_k` | Agreement between runs against k (3, 3.5, 4) for every library, means joined along k | data/quality/agreement.csv |
| `quality/secondary/quality_by_library` | Quality against chemistry (as quality_chembl34) for every library | data/quality/by_run, by_run_earlier_labels |
| `quality/secondary/quality_vs_k` | Quality against k for every library (mean and 95 % interval) | data/quality/by_run, by_run_earlier_labels |
| `quality/secondary/refinement_chembl34` | Quality on ChEMBL34 before and after 10 passes of bblean's recluster refinement | data/refinement/quality_chembl34.csv |
| `quality/secondary/refinement_passes_all` | Quality against the refinement pass (recluster, no threshold increase) for every library, all input orders | data/refinement/quality_<library>.csv |

## quality / extra

| figure | what it shows | data |
|---|---|---|
| `quality/extra/ami_vs_clusters_c35` | AMI with scaffolds and targets against the number of clusters (k = 3, 3.5, 4), one library | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/ami_vs_clusters_chembl34` | AMI with scaffolds and targets against the number of clusters (k = 3, 3.5, 4), one library | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/ami_vs_clusters_ena1m` | AMI with scaffolds and targets against the number of clusters (k = 3, 3.5, 4), one library | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/ami_vs_clusters_np` | AMI with scaffolds and targets against the number of clusters (k = 3, 3.5, 4), one library | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/ami_vs_clusters_scu` | AMI with scaffolds and targets against the number of clusters (k = 3, 3.5, 4), one library | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/clusters_by_library` | Number of clusters per implementation and library, k = 3.5 | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/quality_all_scores` | AMI, ARI, homogeneity, completeness and V-measure against scaffolds and targets, every library | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/quality_vs_k_c35` | Quality against k for one library, one dot per input order | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/quality_vs_k_chembl34` | Quality against k for one library, one dot per input order | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/quality_vs_k_ena1m` | Quality against k for one library, one dot per input order | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/quality_vs_k_np` | Quality against k for one library, one dot per input order | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/quality_vs_k_scu` | Quality against k for one library, one dot per input order | data/quality/by_run, by_run_earlier_labels |
| `quality/extra/refinement_passes_rc0_c35` | Quality against the refinement pass (recluster, no threshold increase), one library, input order 0 | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc0_chembl34` | Quality against the refinement pass (recluster, no threshold increase), one library, input order 0 | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc0_ena1m` | Quality against the refinement pass (recluster, no threshold increase), one library, input order 0 | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc0_np` | Quality against the refinement pass (recluster, no threshold increase), one library, input order 0 | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc0_scu` | Quality against the refinement pass (recluster, no threshold increase), one library, input order 0 | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc1_all` | Quality against the refinement pass (recluster, threshold grows by sigma each pass), every library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc1_c35` | Quality against the refinement pass (recluster, threshold grows by sigma each pass), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc1_chembl34` | Quality against the refinement pass (recluster, threshold grows by sigma each pass), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc1_ena1m` | Quality against the refinement pass (recluster, threshold grows by sigma each pass), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc1_np` | Quality against the refinement pass (recluster, threshold grows by sigma each pass), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_rc1_scu` | Quality against the refinement pass (recluster, threshold grows by sigma each pass), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_ref_all` | Quality against the refinement pass (refine the largest cluster), every library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_ref_c35` | Quality against the refinement pass (refine the largest cluster), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_ref_chembl34` | Quality against the refinement pass (refine the largest cluster), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_ref_ena1m` | Quality against the refinement pass (refine the largest cluster), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_ref_np` | Quality against the refinement pass (refine the largest cluster), one library | data/refinement/quality_<library>.csv |
| `quality/extra/refinement_passes_ref_scu` | Quality against the refinement pass (refine the largest cluster), one library | data/refinement/quality_<library>.csv |
| `quality/extra/top20_clusters_chembl34_k3` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_chembl34_k3.5` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_chembl34_k4` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_ena1m_k3` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_ena1m_k3.5` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_ena1m_k4` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_scu_k3` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_scu_k3.5` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/top20_clusters_scu_k4` | Sizes and unique scaffolds of the 20 largest clusters and clusters above each size, one library and k | data/quality/top_clusters.csv, size_bins.csv |
| `quality/extra/topk_c35` | Quality of the largest clusters (5, 20, 100, 1000 largest), one library | data/quality/by_run/topk_<library>.csv |
| `quality/extra/topk_chembl34` | Quality of the largest clusters (5, 20, 100, 1000 largest), one library | data/quality/by_run/topk_<library>.csv |
| `quality/extra/topk_ena1m` | Quality of the largest clusters (5, 20, 100, 1000 largest), one library | data/quality/by_run/topk_<library>.csv |
| `quality/extra/topk_np` | Quality of the largest clusters (5, 20, 100, 1000 largest), one library | data/quality/by_run/topk_<library>.csv |
| `quality/extra/topk_scu` | Quality of the largest clusters (5, 20, 100, 1000 largest), one library | data/quality/by_run/topk_<library>.csv |

## timing / primary

| figure | what it shows | data |
|---|---|---|
| `timing/primary/time_vs_size_zinc20` | Clustering time against size, ZINC20 subsets of 10M to 101M molecules | data/timing/timings.csv |
| `timing/primary/time_vs_workers_enamine10m` | Clustering time against CPU threads / processes on Enamine 10M, linear and log-log axes | data/timing/timings.csv |

## timing / secondary

| figure | what it shows | data |
|---|---|---|
| `timing/secondary/memory_zinc20` | Peak memory against size, ZINC20 subsets | data/timing/timings.csv |
| `timing/secondary/time_vs_workers_chembl34` | Clustering time against CPU threads / processes on ChEMBL34 | data/timing/timings.csv |

## timing / extra

(none)

## other / primary

| figure | what it shows | data |
|---|---|---|
| `other/primary/zinc22_cluster_sizes` | Sizes of the 10,000 largest clusters of the 12.16 billion molecule ZINC22 run (nvMolKit batched) | data/zinc22/largest_cluster_sizes.csv |

## other / secondary

| figure | what it shows | data |
|---|---|---|
| `other/secondary/alg_schematic` | Schematic of one round of the batched algorithm (no data) | none |
| `other/secondary/moleculeace_by_set` | MoleculeACE, threshold 0.65: clusters and three clustering indices, mean over the 30 sets | data/moleculeace/moleculeace_indices.csv |

## other / extra

| figure | what it shows | data |
|---|---|---|
| `other/extra/bbl_serial_vs_parallel_top20_chembl34` | Sizes and iSIM of the 20 largest clusters, bblean serial against the others, one library | data/quality/top_clusters.csv |
| `other/extra/bbl_serial_vs_parallel_top20_ena1m` | Sizes and iSIM of the 20 largest clusters, bblean serial against the others, one library | data/quality/top_clusters.csv |
| `other/extra/bbl_serial_vs_parallel_top20_scu` | Sizes and iSIM of the 20 largest clusters, bblean serial against the others, one library | data/quality/top_clusters.csv |
| `other/extra/moleculeace_clusters_vs_threshold_CHEMBL262_Ki` | Clusters with at least 1, 2, 3, 5, 10 members against threshold, one MoleculeACE set | data/moleculeace/moleculeace_indices.csv |
| `other/extra/moleculeace_indices_vs_threshold_CHEMBL233_Ki` | Calinski-Harabasz, Davies-Bouldin and Dunn index against threshold, one MoleculeACE set | data/moleculeace/moleculeace_indices.csv |
| `other/extra/zinc22_vs_libraries` | The ZINC22 run next to three smaller libraries (nvMolKit batched): whole clustering and largest clusters | data/zinc22/zinc22_overall.csv, zinc22_topk.csv; data/quality/by_run, by_run_earlier_labels |

There are two .pth files with labelled source (MRI-T1) and unlabelled target (MRI-T2) data. 


== 'HackathonDA_CHAOS_T1_source.pth' ==
contains 20 manually labelled 3D scans "train_source_data" and 20 automatically labelled ones "pseudo_source_data" with approx. 160-256 x 160-256 spatial dimension and **two** channels (you may choose to ignore the second T1 channel). The dictionary contents are a list of 20 tensors each with a size of 3 x N x H x W, where the last entry of the 0th dimension contains the organ segmentation. There are usually 20-40 slices with various thicknesses in one stack, so it might be easiest to treat them as individual 2D data points.
 
{'train_source_data':train_source_data,'pseudo_source_data':pseudo_source_data,'train_source_names':train_source_names,'pseudo_source_names':pseudo_source_names}

== 'HackathonDA_CHAOS_T2.pth' ==
contains 26 unlabelled target training MRI T2 (3D) scans of corresponding patients but acquired in a different session (so assumed to be not in alignment!). The dimensions are similar as for T1 but now there is only on channel: 1 x N x H x W. There are also 7 unlabelled test scans that will serve as benchmark for our competition. We also provide 7 pseudo-labelled validation T2 scans with dimensions 2 x N x H x W - that should not be used for training but only to track the performance of your unsupervised algorithms.

{'train_data':train_data,'val_data':val_data,'test_data':test_data,'train_names':train_names,'val_names':val_names,'test_names':test_names}
 

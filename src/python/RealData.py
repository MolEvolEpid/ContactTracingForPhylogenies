"""
RealData.py
Michael Kupperman

Handle data as if it was real.

"""
import os
import math
import csv

from typing import Union
from collections import namedtuple

import numpy as np
from scipy.spatial.distance import squareform
from scipy.io import savemat
import scipy.io as scio
import scipy.cluster.hierarchy as hclust

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Define a named tuple to export names
PredictResults = namedtuple('PredictResults', ['person_predictions', 'idx_map', 'label_dict', 'scores'])


class RealData(object):
    def __init__(self, source, source_type, scale_factor=1, shuffle=False, include_only=None, expected_names=None):
        """ Import a dataset styled after real data. Not all fields will be populated.
        Args:
            source_type: One of 'real', 'real_csv', 'synthetic', or '_dict'.
        keyword `source_type` indicates which values to unpack from `source_type`.
        Source type '_dict' should only be used for dictionary inputs and internal methods.
        `randomize_data` is currently only implemented for synthetic data with parquet loading.

        include_only is only considered for source_type of real_csv. All row labels not in include_only will be
        removed, and the matrix appropriately modified.
        """
        # Always added
        self.matrices = None

        # Added in real data and some synthetic data
        self.row_labels = None  # Labels for each row of the matrix
        self.row_names = None  # List of names of elements in each matrix (by row)

        # Only added for synthetic data
        self.labels = None
        self.shuffle = None
        self.NPop = None
        self.times = None
        self.R0 = None
        self.matrix_store = {}
        self.row_names_store = {}
        self.cluster_size = None
        self.sequence_length = None
        self.active_case = None
        self.cases = None
        self.categorical_labels = None

        # For cluster data
        self.num_clusters = None
        self.size_clusters = None


        self.source_type = source_type
        if source_type == 'real':
            self.load_real_data(source, scale_factor=scale_factor)
        elif source_type == 'real_csv':
            self.load_csv(source_file=source, scale_factor=scale_factor, include_only=include_only)
            self.source_type = 'real'  # we no longer care about the CSV import
        elif source_type == 'synthetic' or source_type == 'synth':
            self.load_synth_data(source)
            self.source_type = 'synth'
        elif source_type == '_dict':
            self.load_from_dict(source)
        elif source_type == 'parquet':
            self.load_from_parquet(source, shuffle=shuffle, scale_factor=scale_factor, expected_names=expected_names)
        else:
            raise NotImplementedError(f'The source type {source_type} is not implemented')

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        string = f"RealData object with {self.matrices.shape[0]} matrices of shape {self.matrices.shape[1:]}\n"
        string += f"Has R0: {self.R0 is not None}\nHas NPop: {self.NPop is not None}\nHas mu: {self.mu is not None}\n"
        string += f"Has times: {self.times is not None}\nHas contact tracing p: {self.contact_tracing_p is not None}\n"
        string += f"Has sequence length: {self.sequence_length is not None}\n"
        string += f"Has cluster size: {self.cluster_size is not None}\nHas num clusters: {self.num_clusters is not None}\n"
        string += f"Has shuffle maps: {self.shuffle is not None}\n"
        string += f"Has multiple matrices: {len(self.matrix_store) > 0}, with keys: {list(self.matrix_store.keys())}\n"
        string += f"Has multiple labels: {len(self.matrix_store) > 0} with keys: {list(self.row_names_store.keys())}\n"
        string += f"Current active case: '{self.active_case}'\n"
        string += f"Registered cases: {self.cases}\n"
        string += f"Categorical labels set: {self.categorical_labels is not None}\nRow labels set: {self.row_labels is not None}\n"

        return string

    def load_from_parquet(self, source_file, expected_names = None,
                          shuffle=False, scale_factor=1):
        """ Load data from a parquet file. This is a special case of synthetic data, but we want to
        keep it separate for now as it presents additional structure within the data set not present
        in the V1 synthetic data that uses .mat files.
        """

        if expected_names is None:
            expected_names = ["matrix_trans", "matrix_phylo", "matrix_seqs"]
        import pandas as pd
        df = pd.read_parquet(source_file)
        if shuffle:
            subset = df[expected_names[0]]
            mat_shape = int(math.sqrt(subset[0].shape[0]))
            sort_size = (len(subset), mat_shape, mat_shape,)
            print(sort_size)
            self.shuffle = get_shuffle_map(sort_size)

        for name in expected_names:
            res = df[name].apply(lambda x: np.array(x.reshape(int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0])))) * scale_factor )
            matrix_stack = np.stack(res.values.tolist())
            if shuffle:
                print(matrix_stack.shape)
                matrix_stack = shuffle_matrix_stack(matrix_stack=matrix_stack, shuffle_map=self.shuffle)
            self.matrix_store[name] = matrix_stack
            # Unpack labels
            row_label_list = []
            for record in [x.tolist() for x in df[name + "_names"].to_numpy().tolist()]:
                # Convert strings to integera
                row_label_list.append([int(x) for x in record])
            self.row_names_store[name] = np.asarray(row_label_list, dtype=int)


        # Set matrix_seqs active
        self.active_case = expected_names[0]
        self.matrices = self.matrix_store[self.active_case]
        self.row_names = self.row_names_store[self.active_case]
        # unpack remaining columns we can just "read off" the dataframe

        # Get tree summary statistics if available

        # EBL/IBL ratios: "ei_ratio"
        # time scaled trees
        if "ei_ratio_phylo" in df.columns:
            self.ei_ratio_phylo = df["ei_ratio_phylo"].to_numpy()
        if "ei_ratio_trans" in df.columns:
            self.ei_ratio_trans = df["ei_ratio_trans"].to_numpy()
        # Mutation-length idealized trees have _mu suffix
        if "ei_ratio_phylo_mu" in df.columns:
            self.ei_ratio_phylo_mu = df["ei_ratio_phylo_mu"].to_numpy()
        if "ei_ratio_trans_mu" in df.columns:
            self.ei_ratio_trans_mu = df["ei_ratio_trans_mu"].to_numpy()

        # Sackin indices
        if "sackin_trans" in df.columns:
            self.sackin_trans = df["sackin_trans"].to_numpy()
        if "sackin_phylo" in df.columns:
            self.sackin_phylo = df["sackin_phylo"].to_numpy()
        if "sackin_trans_mu" in df.columns:
            self.sackin_trans_mu = df["sackin_trans_mu"].to_numpy()
        if "sackin_phylo_mu" in df.columns:
            self.sackin_phylo_mu = df["sackin_phylo_mu"].to_numpy()

        # Number of cherries
        if "cherries_trans" in df.columns:
            self.cherries_trans = df["cherries_trans"].to_numpy()
        if "cherries_phylo" in df.columns:
            self.cherries_phylo = df["cherries_phylo"].to_numpy()
        if "cherries_trans_mu" in df.columns:
            self.cherries_trans_mu = df["cherries_trans_mu"].to_numpy()
        if "cherries_phylo_mu" in df.columns:
            self.cherries_phylo_mu = df["cherries_phylo_mu"].to_numpy()


        # Get R0
        self.R0 = df["R0"].to_numpy()
        # Get NPop
        self.NPop = df["maximum_population_target"].to_numpy()
        self.mu = df["mutation_rate"].to_numpy()
        self.sequence_length = df["sequence_length"].to_numpy()
        self.times = df["total_steps_after_exp_phase"].to_numpy() / 12  # From months to years
        self.contact_tracing_p = df["contact_tracing_discovery_probability"].to_numpy()
        self.cases = expected_names.copy()  # Don't use the variable bound in the prototype

        # Setup labels: self.row_labels, self.categorical_labels
        self.encode_labels_using_time()  # use times as labels

    def load_real_data(self, source_file, scale_factor):
        """Real data offers limited info, we only want to load in the known fields.


        Args:
            source_file: String path to data `.mat` file.
            scale_factor: rescale matrix by factor, useful for seq length/evolutionary distance conversions

        Returns:
            None

        """

        data = scio.loadmat(source_file)
        self.matrices = data['matrices']
        self.matrices = self.matrices * scale_factor
        self.row_labels = data['row_labels']

    def load_csv(self, source_file, scale_factor, flip_table=True, include_only=None):
        if flip_table:
            a = zip(*csv.reader(open(source_file, "r")))
            csv.writer(open('.tmp.csv', 'w')).writerows(a)
            source_file = '.tmp.csv'
        self.matrices = np.loadtxt(source_file, delimiter=',', skiprows=1) * scale_factor
        with open(source_file) as handle:
            reader = csv.reader(handle)
            self.row_labels = next(reader)
        if flip_table:
            # clean up the temp file
            os.remove(source_file)
        if include_only is not None:
            try:
                assert (hasattr(include_only, '__iter__')), 'The option for include_only must be iterable'
            except AssertionError:
                print('The provided include_only is not iterable. Continuing execution.')
                return True  # exit safely
            remove_indexes = []
            for idx, key in enumerate(self.row_labels):
                if key not in include_only:
                    remove_indexes.append(idx)
            for index in reversed(remove_indexes):
                del self.row_labels[index]  # remove it from the list

            # Numpy can do this without the loop
            remove_indexes = np.asarray(remove_indexes, dtype=int)
            np.delete(self.matrices, remove_indexes, axis=0)
            np.delete(self.matrices, remove_indexes, axis=1)

    def load_synth_data(self, source_file):
        """
        synthetic data offers generation metadata which we want to store.
        From R:
          clusters=number_of_clusters,
          cluster_size=cluster_sample_size,
          matrices = data,    # the large composite matrices, in a 3d array
          labels = labels,    # labels for each person
          shuffle = shuffle,  # store the shuffle maps
          NPop= NPop          # store the simulation pop size for each person
          Some data objects may come with an additional R0 attribute.
        """

        data = scio.loadmat(source_file)

        self.matrices = data['matrices']
        self.labels = data['labels']
        self.shuffle = data['shuffle']
        self.NPop = data['NPop']
        self.cluster_size = data['cluster_size']
        self.num_clusters = data['clusters']
        self.times = data['times']
        print(data.keys())
        if 'R0' in data.keys():
            print('Adding R0')
            self.R0 = data['R0']

    def load_from_dict(self, data):
        """ Load data from a dictionary. Useful for generating subsets from larger sets. """
        self.matrices = data['matrices']
        self.row_labels = data['row_labels']
        self.source_type = 'real'  # We got years, so real data

    def _real_to_dict(self, keys=None):
        """ Return a copy of the matrices and row labels indicated by keys.
        If keys is None (default), all rows and row labels are returned. Otherwise, provide an index to select.

        See `load_from_dict` method above.
        """
        if keys is None:  # default to selecting all of the data
            keys = np.arange(self.matrices.shape[0])
        return {'matrices': self.matrices[keys, :][:, keys].copy(),
                'row_labels': np.asarray(self.row_labels)[keys].copy().tolist()}  # cast list -> np.ndarray -> list

    def encode_labels_using_time(self):
        """ Encode labels as integers. Construct new encoders. """
        # Get the unique times in ascending order
        self.times = np.asarray(self.times, dtype=int)
        unique_times = np.unique(self.times)
        unique_times.sort()
        print(unique_times)

        self.labelEncoder = LabelEncoder()
        self.oneHotEncoder = OneHotEncoder(sparse=False)
        u_labels = self.labelEncoder.fit_transform(unique_times)
        print(unique_times, u_labels)
        self.oneHotEncoder.fit(u_labels.reshape(-1, 1))
        flat_labels, ohe_labels = self.transform_to_labels(self.times)
        self.row_labels = flat_labels
        self.categorical_labels = ohe_labels



    def transform_to_labels(self, times):
        """ Apply transformations to data """
        times = np.asarray(times, dtype=int)
        flat_labels = self.labelEncoder.transform(times)
        ohe_labels = self.oneHotEncoder.transform(flat_labels.reshape(-1, 1))
        return flat_labels, ohe_labels

    def set_active(self, active):
        """ Set the active matrix. """
        if active in self.cases:
            self.active_case = active
            self.matrices = self.matrix_store[active]
            self.row_names = self.row_names_store[active]
        else:
            raise KeyError(f"The requested active case {active} is not in the active list {self.cases}")


    def to_training_format(self, save_filename, labels, nexamples, size, save=False, sampler='uniform'):
        ddict = {}
        label = np.ones(shape=nexamples) * labels
        matrices = np.zeros((nexamples, size, size))
        for index in range(nexamples):
            if sampler == 'uniform':
                keys = np.random.choice(self.matrices.shape[0], size, replace=False)
            elif sampler == 'density':
                center = np.random.choice(self.matrices.shape[0], 1, replace=False)  # choose the "center"
                width = np.minimum(self.matrices.shape[0],
                                   size - 1 + np.random.geometric(p=0.05))  # mean of geo(0.1) = 1/0.1 = 10
                # 0.05 => mean is 20
                # Ensure that we never ask for more data than is present.
                # Minimum sample size should be (size) + 1, so we always have some randomness
                distances = self.matrices[center, :]  # get all the other distances
                dist_map = np.argsort(distances)
                keys = dist_map[0:width]  # get the leading width locations
                keys = keys.flatten()
                keys = np.random.choice(keys, size, replace=False)  # now sample the interval we built
            elif sampler == 'uniform_density':
                center = np.random.choice(self.matrices.shape[0], 1, replace=False)  # choose the "center"
                width = np.minimum(self.matrices.shape[0],
                                   size + np.random.choice((self.matrices.shape[0] - size - 5) // 2, 1))
                # 5 here is a safeguard
                width = width.item()
                distances = self.matrices[center, :]  # get all the other distances
                dist_map = np.argsort(distances)
                keys = dist_map[0:2 * width]  # get the leading 2*width locations
                keys = keys.flatten()
                keys = np.random.choice(keys, size, replace=False)  # now sample the interval we built

            else:
                raise NotImplementedError

            matrices[index, :, :] = self.matrices[keys, :][:, keys]
        ddict['matrices'] = matrices
        ddict['labels'] = label.copy()
        ddict['NPop'] = np.zeros_like(label)
        ddict['values'] = np.zeros((nexamples, 2))
        ddict['indexes'] = np.zeros((nexamples, 2))
        ddict['indexes'][:, 0] = label
        ddict['values'][:, 0] = label
        if save:
            savemat(save_filename, ddict)
        else:
            return ddict

    def predict_by_year(self, model, window_size, choice_method, rule='forwards', matrix=1,
                        cluster_method='None', second_clustering_method='None', return_map=False):
        """ Compute predictions on data by year using model.predict() method.
        Returns dictionaries for predictions, year-subset source data, and sort aggregates
        """

        data_subset, sort_dict = self.subset_by_year(min_size=window_size, rule=rule)
        pred_dict = {}
        rd_dict = {}
        for key in data_subset.keys():
            rd_tmp = RealData(source=data_subset[key], source_type='_dict')
            rd_dict[key] = rd_tmp
            preds = rd_tmp.predict(model=model, window_size=window_size, choice_method=choice_method,
                                   matrix=matrix, cluster_method=cluster_method,
                                   second_clustering_method=second_clustering_method, return_map=return_map)
            pred_dict[key] = preds
        return pred_dict, rd_dict, sort_dict

    def subset_by_year(self, min_size=15, rule='forwards'):
        """ generate a dict of RealData with infections sorted by year.
        Only return images with more than `min_size` elements. Specify
        `rule=forward` to join samples that are less than min_size
        with the next year's sample. see `join_dict_by_size` for details on the joining.
        """
        assert type(min_size) is int
        assert min_size > 1, 'An image cannot have less than 1 infection'
        year_dict = self._filter_by_year()
        data = self._real_to_dict()
        new_subset = {}
        # Filter
        if rule == 'forwards':
            year_dict, sort_dict = join_dict_by_size(min_size=min_size, data=year_dict, rule='forwards')
        elif rule == 'backwards':
            year_dict, sort_dict = join_dict_by_size(min_size=min_size, data=year_dict, rule='backwards')
        elif rule == 'None' or type(rule) is None:
            sort_dict = {key: [key] for key in year_dict.keys()}
        else:
            raise ValueError(f'Rule {rule} was not a recognized option')
        # Build images
        for year in year_dict.keys():  # loop over each year
            indiv = np.asarray(year_dict[year])  # sort indexes and cast to numpy array
            num_indiv = indiv.shape
            new_im = data['matrices'].copy()  # fresh copy
            # Subset by row and by column, one at a time
            new_im = new_im[indiv, :]
            new_im = new_im[:, indiv]
            # print(new_im)
            new_labels = data['row_labels'][indiv]
            new_subset[year] = {'matrices': new_im, 'row_labels': new_labels}
        return new_subset, sort_dict

    def _filter_by_year(self):
        """ Generate a dict of infections by year.

        A utility method for filtering the data by year. Enables predictions based on each year, if sample order is
        not known.

        """
        assert self.source_type == 'real', 'Cannot filter by year without labels present'
        years = []
        # Build a list of sample years
        for label in self.row_labels.ravel():
            tokens = np.char.split(label, sep='.').item()
            year_set = False
            # noinspection PyTypeChecker
            for token in tokens:
                try:
                    if int(token) > 2100 or int(token) < 1980:
                        raise ValueError  # break the try
                    year = int(token)

                    year_set = True
                except ValueError:
                    pass  # do nothing
                if not year_set:
                    year = 0  # assign it to year 0,
            # noinspection PyUnboundLocalVariable
            years.append(year)
        # Generate a dict to store results
        samples_per_year = {key: [] for key in years}
        for idx in range(len(years)):
            # make a list of infection indices by year
            samples_per_year[years[idx]].append(idx)
        for key in samples_per_year.keys():
            samples_per_year[key].sort()
        return samples_per_year

    def _visualize_matrix(self, index, cluster_method='None', preds=None, gt=None,
                          fig=None, ax=None, create_cbar=False):
        """ Plot the matrix.

        Options support clustering & prediction overlays, ground truth values, and to create a new
        colorbar axis or use the default constructor.
        """

        if ax is None and fig is None:
            fig, ax = plt.subplots(1)
        image = self._get_image(index, cluster_method=cluster_method, return_map=False)
        # image = np.squeeze(image[0])
        vmax = np.max(image)
        image = np.squeeze(image)
        main_IM = ax.imshow(image, cmap='plasma', vmin=0, vmax=vmax, interpolation='nearest')
        if create_cbar:
            # Create the
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.1)
            plt.colorbar(main_IM, cax=cax)
        else:
            fig.colorbar(main_IM, ax=ax)
        if preds is not None:
            overlay_preds(axes=ax, preds=preds, edge_color='y', face_color='y', ls=None)
        if gt is not None:
            print('Adding ground truth values to overlay')
            overlay_preds(axes=ax, preds=gt, edge_color='r', face_color='r', ls='--', alpha=0.4)
        return fig, ax

    def show(self, index, cluster_method='None', second_clust_method='None', model=None, fig_obj=None,
             create_cbar=False, window_size=None,
             overlay_preds=False, assign_method='window',
             highlight=0, choice_method='argmax', ax=None,
             add_gt=False, **kwargs):
        """
        Return a pyplot figure and axes for matrices[index].

        Args:
            index:
            fig_obj:
            second_clust_method:
            choice_method:
            create_cbar:
            cluster_method: string, method for first clustering.
            model: An object with a predict method. See predict method for this class for more information
            window_size: integer, see class predict method for more information
            overlay_preds: bool, add boxes to plot where pred=highlight
            highlight: int, Model label to highlight.
            ax (pyplot axis): Optionally attach the plot to a specified axis.
            add_gt: bool, add ground-truth labels to predictive overlay.
        """

        gt = None
        preds = None
        if overlay_preds:
            assert model is not None, 'A model must be provided'
            assert highlight is not None and type(highlight) is int, 'highlight must be an int'
            assert window_size is not None, 'Window size must be specified'
            prediction_output = self.predict(model=model, window_size=window_size, cluster_method=cluster_method,
                                             matrix=index, assign_method=assign_method,
                                             second_clustering_method=second_clust_method, choice_method=choice_method,
                                             return_map=True, **kwargs)
            preds, idx_map = prediction_output.person_predictions, prediction_output.idx_map
            # scores = np.mean([prediction_output.scores])
            confidence = [np.mean(v) for v in prediction_output.scores]
            preds = preds[idx_map]  # We want the sorted predictions, not the initial inputs
            preds = np.equal(preds, highlight)  # only pass the logical indices where we want boxes
            if add_gt:
                # Compute the correct overlay based on known labels.
                # Will fail if source_type='real' at data import
                gt = self.labels[index][idx_map] == (highlight + 1)
                print('gt is:', gt)
        return self._visualize_matrix(index, cluster_method=cluster_method, preds=preds, ax=ax, gt=gt,
                                      fig=fig_obj, create_cbar=create_cbar)

    def lineplot(self, index: int, models: list = None, window_sizes: list[Union[int, list]] = None,
                 choice_method: str = 'argmax', cluster_method: str = 'HC', second_cluster_method: str = 'None',
                 cmap_key: str = 'Set2', show_mat: bool = True):
        """ Generate a 1d line plot showing "alignment"-style predictions.
        """
        if window_sizes is None:
            window_sizes = []
        if models is None:
            models = []

        preds_list = []
        if type(models) is not list:
            models = [models]
        if type(window_sizes) is not list:
            window_sizes = [window_sizes]
        for model, window_size in zip(models, window_sizes):
            predict_return = self.predict(model=model, window_size=window_size, cluster_method=cluster_method,
                                          matrix=index,
                                          second_clustering_method=second_cluster_method, choice_method=choice_method,
                                          return_map=True)
            preds, idx_map = predict_return.person_predictions, predict_return.idx_map
            preds = preds[idx_map]  # We want the sorted predictions, not the initial inputs
            preds_list.append(preds)
        gt = None
        if self.source_type == 'synth':
            gt = self.labels[index, idx_map] - 1

        fig, ax, heatmap = map_alignment(gt=gt, preds=preds_list, cmap_key=cmap_key, add_second_axis=show_mat)
        fig.suptitle(f'Image {index} with {self.matrices.shape[1]} Infections')
        return fig, ax

    def lambdafy(self, fun, window_size, cluster_method='None'):
        """
        Apply a callable to each matrix after applying the cluster method. Return the result.

        To support computing a statistic over the collection of matrices.
        Args:
            window_size: int
            fun: callable
            cluster_method: string, one of "HC", "OLO", "None"

        Returns:
        """

        assert callable(fun), "Provided function is not callable"
        num_sub_images = self.matrices.shape[1] - window_size + 1
        image_full, idx_map = self._get_image(index=0, cluster_method=cluster_method, return_map=True)
        image_stack = stride_image_into_tensor(image_full, window_size=window_size, start=0, batch_size=num_sub_images)
        image_stack = np.squeeze(image_stack)
        outputs = []
        for index in range(image_stack.shape[0]):
            # loop over each image and apply function
            outputs.append(fun(image_stack[index, :, :]))
        return outputs

    def _get_image(self, index, cluster_method='None', return_map=False):
        """
        Return a tuple of the matrix specified by index coersed
        into the correct shape and apply the specified cluster_method
        and also returns the ordering map computed.
        """

        dim = len(self.matrices.shape)
        # We need to copy here to avoid side effects
        if dim == 2:
            matrix = np.expand_dims(self.matrices.copy(), 2)  # image is 2d, flat
        else:
            matrix = np.expand_dims(self.matrices[index, :, :].copy(), 2)

        idx, matrix = self._sort(index=index, matrix=matrix, cluster_method=cluster_method)

        if return_map:
            return matrix, idx
        else:
            return matrix

    @staticmethod
    def _sort(index, matrix, cluster_method):
        """ A wrapper to determine the sorting method that should be applied. Sort is an external function """
        if cluster_method == 'None':
            idx = np.asarray([index for index in range(matrix.shape[1])])
        elif cluster_method == 'OLO':
            matrix, idx = leaf_order_optimize(matrix)
        elif cluster_method == 'HC':
            matrix, idx = hc(matrix)
        else:
            raise NotImplementedError(f'Cluster method {cluster_method} was not recoginzed')
        return idx, matrix

    @staticmethod
    def batch_sort(image, method):
        """ Apply sort to each layer of image[idx,:,:] """
        image_new = np.empty(shape=image.shape)
        orderings = np.empty(shape=(image.shape[0], image.shape[1]), dtype=np.int)
        # switch on method
        if method == 'None':
            method = no_sort
        elif method == 'OLO':
            method = leaf_order_optimize
        elif method == 'HC':
            method = hc
        else:
            raise ValueError('method not recognized')

        # Apply the method to each layer
        for layer_id in range(image.shape[0]):
            im_new, order = method(image[layer_id, :, :, 0])
            im_new = im_new[:, :, np.newaxis]
            image_new[layer_id, :, :, :] = im_new
            orderings[layer_id, :] = order

        return image_new, orderings

    def get_ordered_labels(self, index, cluster_method):
        image_full, idx_map = self._get_image(index=index, cluster_method=cluster_method, return_map=True)
        labels = self.labels[index, :]
        labels = labels[idx_map]
        return labels

    def predict(self, model, window_size, choice_method, matrix=1, cluster_method='None', assign_method='window',
                second_clustering_method='None', return_map=False, return_labels=False, **kwargs):
        """
        Evaluate the dataset with the `predict` method of input model.

        Predict should return the predicted label, not a probability distribution
        window_size is specific to the model.
        Argument choice_method specified the method to assign final labels for each
        individual.
        Argument matrix specifies the index of the matrix in data.



        return: np.ndarray of predicted label
        """
        if 'sampler_p' not in kwargs:
            kwargs['sampler_p'] = 0.2  # default parameter

        scores = [[] for _ in range(self.matrices.shape[1])]  # per person scores
        image_full, idx_map = self._get_image(index=matrix, cluster_method=cluster_method, return_map=True)
        if assign_method == 'window':
            num_sub_images = self.matrices.shape[1] - window_size + 1
            for index in range(0, num_sub_images, 128):  # step by batches
                batch_size = min(num_sub_images - index, 128)  # ensure last batch doesn't read off array
                image = stride_image_into_tensor(image_full, window_size=window_size, start=index,
                                                 batch_size=batch_size)
                image, idx_map_second = self.batch_sort(image=image, method=second_clustering_method)
                result = model.predict(image)
                for im_idx in range(batch_size):
                    # Loop over multiple images
                    for person in range(window_size):
                        # use idx_map to correctly assign scores
                        scores[idx_map[im_idx + index + idx_map_second[im_idx, person]]].append(
                            result[im_idx].astype(np.int).item(0))
            person_predictions = self.compute_scores(scores, choice_method)
        elif assign_method == 'sampler':
            assert 'windows_per_sample' in kwargs, "Keyword argument 'windows_per_sample' is required"
            assert 'sampler_p' in kwargs, "keyword argument 'sampler_p' is required"
            max_matrix_size = self.matrices.shape[1]
            batch_size = 1028  # could be global, or larger
            batch = np.zeros((batch_size, window_size, window_size))
            final_batch = False  # exit flag for final batch
            if kwargs['windows_per_sample'] < batch_size:
                # The batch sizes we ask for are not sufficiently large to be optimal
                batch_counter = 0
                center_index = 0
                seen_counter = 0
                total_seen = 0
                max_values = kwargs['windows_per_sample'] * max_matrix_size
                centers = np.zeros(batch_size, dtype=int)
                for _ in range(max_values):
                    # Accumulator pattern
                    # print(batch_counter, center_index, seen_counter, total_seen, max_values)
                    batch[batch_counter, :, :] = sampler(center=center_index, max_matrix_size=max_matrix_size,
                                                         sample_size=window_size, matrix=self.matrices,
                                                         geom_parameter=kwargs['sampler_p'])
                    centers[batch_counter] = center_index

                    batch_counter += 1
                    seen_counter += 1
                    total_seen += 1
                    if total_seen == max_values:
                        # Truncate the final batch, there's no more data
                        batch = batch[:batch_counter, :, :]
                        centers = centers[:batch_counter]
                        final_batch = True

                    if batch_counter == batch_size or final_batch:
                        # We filled the batch, send it to the model
                        # print('Sending a batch to the model predict method')
                        batch_counter = 0  # reset the counter
                        preds = model.predict(batch)
                        # Pass out the results
                        # print(scores)
                        for person, pred in zip(centers, preds.astype(int)):
                            scores[person].append(pred)

                    if seen_counter == kwargs['windows_per_sample']:
                        center_index += 1  # go to next sample
                        seen_counter = 0  # reset counter

            else:  # default to 1 batch per sample
                for person in range(self.matrices.shape[1]):  # loop over the matrix
                    batch = np.zeros((kwargs['windows_per_sample'], window_size, window_size))
                    for target_sample in range(kwargs['windows_per_sample']):
                        batch[target_sample, :, :] = sampler(center=target_sample, max_matrix_size=max_matrix_size,
                                                             sample_size=window_size, matrix=self.matrices)
                    result = model.predict(batch)
                    scores[idx_map[person]].extend(result.tolist())
            # print(scores)
            person_predictions = self.compute_scores(scores, choice_method)

        else:
            raise ValueError(f'Option {assign_method=} was not recognized')

        if self.source_type == 'real':
            active = [self.row_labels[row_index] for row_index in range(len(self.row_labels)) if
                      person_predictions[row_index] == 0]
        label_dict = {0: [], 1: [], 2: []}  # Our case only has ~3~ 2
        if return_labels:
            # Make lists of labels
            for row_index in range(len(self.row_labels)):
                label_dict[person_predictions[row_index]].append(self.row_labels[row_index].item().item())
                # double item does the correct access through the array of arrays... yikes numpy
        return_type = PredictResults(person_predictions=person_predictions,
                                     idx_map=idx_map, label_dict=label_dict, scores=scores)
        return return_type

        # if return_labels:
        #     if return_map:
        #         #noinspection PyUnboundLocalVariable
        #         return person_predictions, idx_map, label_dict
        #     else:
        #         #noinspection PyUnboundLocalVariable
        #         return person_predictions, label_dict
        # else:
        #     if return_map:
        #         #noinspection PyUnboundLocalVariable
        #         return person_predictions, idx_map
        #     else:
        #         return person_predictions

    def evaluate(self, model, window_size, choice_method, matrix=1, cluster_method='None'):
        """
        Evaluate the performance of a model and choice method on the dataset.
        This method should be called only on synthetic data when true labels are known.

        Returns the accuracy, predictions, and associated labels.
        """
        if type(matrix) is int:
            matrix = [matrix]
        acc_vals = list()
        pred_list = list()
        label_list = list()
        for matrix_id in matrix:
            predResult = self.predict(model=model, window_size=window_size,
                                      choice_method=choice_method, matrix=matrix_id,
                                      cluster_method=cluster_method)
            predictions = predResult.person_predictions
            bool_filter = predictions != -1
            predictions_filtered = predictions[bool_filter]
            acc_vals.append(
                np.sum(np.equal(predictions_filtered, self.labels[matrix_id, bool_filter] - 1)) / np.sum(bool_filter))
            pred_list.append(predictions_filtered)
            label_list.append(self.labels[matrix_id, bool_filter] - 1)
        acc_np = np.array(acc_vals)
        print(
            f'Average score: {np.mean(acc_np)}\nmax score: {np.amax(acc_np)}\nmin score: {np.amin(acc_np)}\n SD: {np.std(acc_np)}\nmedian score: {np.median(acc_np)}')
        return acc_np, pred_list, label_list

    def compute_scores(self, scores, choice_method):
        """ A switch statement to match choice_method, scores are passed through"""
        if choice_method == 'inclusive' or choice_method == 'strict':
            return self._score_strict(scores)
        elif choice_method == 'argmax':
            return self._score_argmax(scores)
        elif choice_method == 'median':
            return self._score_median(scores)
        else:
            raise NotImplementedError(f'The choice method {choice_method} is not implemented')

    def _score_strict(self, scores):
        """
        Assign the smallest score for each person to each person
        """
        return score_strict_(scores)

    def _score_argmax(self, scores):
        """
        Assign the smallest score for each person to each person
        """
        return score_argmax_(scores)

    def _score_median(self, scores):
        """
        Assign the smallest score for each person to each person
        """
        people_scores = np.zeros(shape=(len(scores)))
        for index, person_scores in enumerate(scores):
            person_scores = [value for value in scores[index] if value != None]
            x = np.rint(np.asarray(person_scores)).flatten().astype('int8')
            # print(self.labels[0, index] - 1, ' - ', x)
            # bc = np.bincount(x)
            people_scores[index] = np.median(x)
        return people_scores


def no_sort(matrix):
    """ A simple method to perform an identity transform."""
    order = np.array([val for val in range(matrix.shape[1])])
    return matrix, order


def leaf_order_optimize(matrix, method='ward'):
    distvec = squareform(matrix[:, :, 0])
    linkage_map = hclust.linkage(distvec, method=method)
    optimal_linkage_map = hclust.optimal_leaf_ordering(Z=linkage_map, y=distvec)
    order = hclust.leaves_list(optimal_linkage_map)
    matrix[:, :, 0] = matrix[order, :, 0]
    matrix[:, :, 0] = matrix[:, order, 0]
    return matrix, order


def hc(matrix, method='ward'):
    distvec = squareform(matrix[:, :, 0])
    linkage_map = hclust.linkage(distvec, method=method)
    order = hclust.leaves_list(linkage_map)
    matrix[:, :, 0] = matrix[order, :, 0]
    matrix[:, :, 0] = matrix[:, order, 0]
    return matrix, order


def stride_image_into_tensor(data, start, window_size, batch_size=32):
    """
    Stride window of k x k over d x d x 1 data into n x k x k x 1 tensor.
    """
    tensor = np.empty(shape=(batch_size, window_size, window_size, 1))
    for row in range(batch_size):
        idx = start + row
        end = idx + window_size
        tensor[row, :, :, :] = data[(idx):(end), (idx):(end), :]
    return tensor


def score_strict_(scores):
    people_scores = np.empty(shape=(len(scores)))
    for index, person_scores in enumerate(scores):
        x = np.rint(np.asarray(person_scores)).flatten().astype(np.int)
        people_scores[index] = np.amin(x)
    return people_scores


def score_argmax_(scores):
    people_scores = np.empty(shape=(len(scores)))
    for index in range(len(scores)):
        person_scores = scores[index]  # [value for value in scores[index] if value != None]
        x = np.rint(np.asarray(person_scores)).astype(np.int).flatten()
        bc = np.bincount(x)
        people_scores[index] = np.argmax(bc)
    return people_scores


def overlay_preds(axes, preds, edge_color='w', face_color='w', ls=None, alpha=0.7):
    """ Add tiles to axes where preds=True """
    # parse the list of preds to find the regions where true
    boxes = label_clusters(preds=preds, key=1)
    for start, end in boxes:
        rect = patches.Rectangle((start, start), (end - start), (end - start),
                                 linewidth=1, edgecolor=edge_color, facecolor=face_color, linestyle=ls, alpha=alpha)
        axes.add_patch(rect)
        print(start, end)


def label_clusters(preds, key=1) -> list:
    """ Generate a list of clusters where the prediction matches a key (default is 1)


    Args:
        preds (np.ndarray): List of length (n) of integer predictions
        key: integer to compare predictions against.

    Returns:
        list of tuples of (start, end) for each cluster.

    """
    list_of_patches = []
    start = None
    end = None
    reset_flag = False
    for idx in range(preds.shape[0]):
        if preds[idx] == key:  # add the point to the patch
            if start is None:
                start = idx
        else:  # preds is false and not at the end
            if start is not None:  # We have an open patch
                end = idx - 1  # last position was the last good position
                reset_flag = True
        if idx == (preds.shape[0] - 1):  # catch edge case, last value in matrix
            if start is not None:
                end = idx
                reset_flag = True
        if reset_flag:
            list_of_patches.append((start, end))
            start = None
            end = None
            reset_flag = False

    return list_of_patches


def is_np(array):
    """ Check that the type of the input is a numpy array in a functional pattern"""
    return type(array) is np.ndarray


def map_alignment(gt, preds, cmap_key='viridis', add_second_axis=False):
    """ plot an alignment map of preds against the ground truth `gt`
    gt and rows of preds must be of the same length

    :param gt: array-like of ground truth prediction
    :param preds: list of array-like or array-like predictions
    :param add_second_axis: bool, add a second axis on the right side. Allows for plotting source data elsewhere

    Args:
        cmap_key: color map key. Default is viridis.
        gt: Ground Truth labels
        add_second_axis (bool): Add a second axis to the figure.
    """
    gt_offset = 0
    if gt is not None:
        gt = np.asarray(gt)
        if type(preds) is list:
            # assert all(map(preds, is_np)), 'at least one pred list item is not a numpy array'
            assert all([ar.shape[0] == gt.shape[0] for ar in preds]), 'All arrays must be the same shape'
            preds = np.asarray(preds)
        elif type(preds) is np.ndarray:
            assert preds.shape[1] == gt.shape[0]  # Check the lengths are the same

        nrow = 1 + preds.shape[0]  # gt + samples
        ncol = gt.shape[0]  # alyready checked for same shape
        new_array = np.zeros(shape=(nrow, ncol))
        new_array[0, :] = gt  # Top row is the same
        new_array[1:, :] = preds  # copy the predictions
    else:  # gt is None
        preds = np.asarray(preds)
        new_array = preds.copy()
        gt_offset = -1
        # Copy the array over since we don't need to concatenate
    gridspec_kw = {}
    if add_second_axis:
        gridspec_kw['width_ratios'] = [2, 1]
    fig, ax = plt.subplots(1, 1 + add_second_axis, figsize=(9, 3), gridspec_kw=gridspec_kw)
    if not add_second_axis:
        ax = [ax]
    divider = make_axes_locatable(ax[0])
    cbar_ax = divider.append_axes('right', size='5%', pad=0.1)
    heatmap = ax[0].imshow(new_array, cmap=plt.cm.get_cmap(cmap_key, 3), interpolation='nearest',
                           vmax=2.5, vmin=-0.5, aspect='auto')  # tab10 is good
    for row_id in range(preds.shape[0] + gt_offset):
        # Add a line between the predictions
        y = row_id + 0.5
        ax[0].axhline(y=y, color='k', linewidth=2)
    lt = [0, 1, 2]  # location ticks
    formatting = plt.FuncFormatter(lambda val, loc: lt[loc])
    fig.colorbar(heatmap, cax=cbar_ax, ticks=[0, 1, 2], format=formatting)
    return fig, ax, heatmap


def join_dict_by_size(min_size, data, rule='forwards', verbose=False):
    """ Join entries in data by key order if less than min_size with rules to handle joining order.

    If the final list in data is not joined, it is placed on the previous remaining dataset.
    Return a dict of joined entries and a dict of join records.
    Note that this is knapsack problem and this solution is non-optimal.

    """

    assert type(data) is dict
    assert type(min_size) is int
    print(data)
    keys = list(data.keys())
    key_move_dict = {key: [] for key in keys}
    if rule == 'forwards':
        keys.sort()
    elif rule == 'reverse':
        keys.sort(reverse=True)
    else:
        raise ValueError(f'key order {rule} was not recognized as an allowed case')
    to_move = []
    move_keys = []
    if verbose:
        print('got keys:', keys)
    removed_keys = []  # for debug
    keys_iter = keys.copy()  # iterator doesn't work well when removing keys while iterating
    for key in keys_iter:  # Already sorted
        if verbose:
            print('key:', key)
        # See if we have any points to move forward
        if len(to_move) > 0:
            if verbose:
                print('Attempting to assign keys from temp list')
            # We have some points to move forward
            data[key].extend(to_move)
            key_move_dict[key].extend(move_keys)
            # Clear the temp lists
            to_move.clear()
            move_keys.clear()
        # attempt joining

        # If the cluster is too small, move it
        if len(data[key]) < min_size:  # we can join
            if verbose:
                print('Attempting pop from key', key)
            to_move.extend(data[key])  # set them aside
            move_keys.append(key)
            removed_keys.append(key)
            move_keys.extend(key_move_dict[key])  # move the values we have already clustered
            key_move_dict.pop(key)
            data.pop(key)  # Remove key from dict
            keys.remove(key)  # Remove key from sorted list
            if verbose:
                print('keys remaining', keys)

    # We may have data leftover
    if len(to_move) > 0:  # don't loose data
        target = keys[-1]  # put them on the last list we didn't remove
    for key in keys:
        key_move_dict[key].append(key)
    if verbose:
        print(data)
    return data, key_move_dict


def sampler(center, max_matrix_size, sample_size, matrix, geom_parameter=0.05):
    width = np.minimum(max_matrix_size,
                       sample_size - 1 + np.random.geometric(p=geom_parameter))  # mean of geo(0.2) = 1/0.2 = 5,
    # 0.1 => 1/0.1 = 10
    # Ensure that we never ask for more data than is present.
    # Minimum sample size should be (size) + 1, so we always have some randomness
    distances = matrix[center, :]  # get all the other distances
    dist_map = np.argsort(distances)
    keys = dist_map[0:width]  # get the leading width locations
    keys = keys.flatten()
    keys = np.random.choice(keys, sample_size, replace=False)  # now sample the interval we built
    image = matrix[keys, :][:, keys]

    return image

def shuffle_matrix_stack(matrix_stack, shuffle_map):
    """
    Shuffle the rows and columns of a matrix stack using a
    matrix of indices where each row is a shuffle map.
    """
    assert type(matrix_stack) is np.ndarray
    assert len(matrix_stack.shape) == 3
    assert type(shuffle_map) is np.ndarray
    assert len(shuffle_map.shape) == 2

    new_stack = np.zeros(shape=matrix_stack.shape)
    for i in range(matrix_stack.shape[0]):
        shuffle = shuffle_map[i, :]
        submat = matrix_stack[i, :, :]
        tmp = submat[shuffle, :]
        tmp = tmp[:, shuffle]
        new_stack[i, :, :] = tmp
    return new_stack

def get_shuffle_map(n):
    """ Get a shuffle map for a stack of matrix of size (k,m,m) """

    assert type(n) is tuple
    assert all([type(x) is int for x in n])
    assert len(n) == 3
    assert n[2] == n[1]

    shuffle_map = np.zeros(n[0:2], dtype=int)

    for index in range(n[0]):
        local = np.arange(n[1])
        np.random.shuffle(local)
        shuffle_map[index, :] = local
    return shuffle_map

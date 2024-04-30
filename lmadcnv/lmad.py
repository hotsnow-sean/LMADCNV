import numpy as np
import pandas as pd
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from scipy.spatial.distance import cdist
from skimage.restoration import denoise_tv_chambolle

from .binning import RDdata


def nice_result(result: pd.DataFrame) -> pd.DataFrame:
    """Merge and sort result"""
    if result.empty:
        return result
    result = result.sort_values("start")
    data = [[*result.iloc[0]]]
    for _, [start, end, ty] in result.iloc[1:].iterrows():
        if ty != data[-1][2]:
            if end <= data[-1][1]:
                continue
            data.append([max(start, data[-1][1] + 1), end, ty])
        elif start > data[-1][1] + 1:
            data.append([start, end, ty])
        elif end > data[-1][1]:
            data[-1][1] = end
    return pd.DataFrame(data, columns=["start", "end", "type"])


def get_result(rd: RDdata, labels: np.ndarray, bp_per_bin: int):
    """Generate DataFrame CNV detetion result

    Returns
    -------
    result: pd.DataFrame, title is (start, end, type), type is (gain|loss)
    """
    idx = labels != 0
    test_start = rd.pos[idx] * bp_per_bin + 1
    test_end = test_start + bp_per_bin - 1
    test_type = np.select([labels == 1, labels == -1], ["gain", "loss"], "")
    test_type = test_type[test_type != ""]
    result = pd.DataFrame(
        data=zip(test_start, test_end, test_type), columns=["start", "end", "type"]
    )
    return nice_result(result)


# 1-dimensional case quick computation
# (Rousseeuw, P. J. and Leroy, A. M. (2005) References, in Robust
#  Regression and Outlier Detection, John Wiley & Sons, chapter 4)
def estimate_location_and_mad(data: np.ndarray):
    n_samples = len(data)
    n_support = int(n_samples * 0.6)
    X_sorted = np.sort(np.ravel(data))
    diff = X_sorted[n_support:] - X_sorted[: (n_samples - n_support)]
    halves_start = np.where(diff == np.min(diff))[0]
    # take the middle points' mean to get the robust location estimate
    location = (
        0.5 * (X_sorted[n_support + halves_start] + X_sorted[halves_start]).mean()
    )
    median = X_sorted[n_support // 2 + halves_start[0]]
    abs_diff = np.abs(X_sorted[halves_start[0] : n_support + halves_start[0]] - median)
    mad = np.mean(abs_diff)
    return location, mad


def CBS(rd: np.ndarray) -> list[tuple[int, int]]:
    n_samples = len(rd)

    from rpy2 import robjects
    from rpy2.robjects.packages import importr

    r_base = importr("base")
    r_dnacopy = importr("DNAcopy")
    segs = r_dnacopy.segment(
        r_dnacopy.smooth_CNA(
            r_dnacopy.CNA(
                robjects.vectors.FloatVector(rd),
                r_base.rep(1, n_samples),
                r_base.seq(0, n_samples - 1),
            )
        ),
        verbose=1,
    ).rx2("output")

    seg_idx = []
    for start, end in zip(segs.rx2("loc.start"), segs.rx2("loc.end")):
        # [start, end] -> [start, end)
        seg_idx.append((int(start), int(end) + 1))
    return seg_idx


def fill_invalid_region(data: RDdata, seg_idx: list[tuple[int, int]]):
    rd, pos = list(data.RD), list(data.pos)
    sum_offset = 0
    for seg_id, (start, end) in enumerate(seg_idx):
        cur_offset = 0
        for i in range(start + 1, end):
            if data.pos[i - 1] + 1 >= data.pos[i]:
                continue
            fill_idx = sum_offset + start + cur_offset + i
            # rd[fill_idx:fill_idx] = np.linspace(
            #     data.RD[i - 1], data.RD[i], data.pos[i] - data.pos[i - 1] - 1
            # )
            rd[fill_idx:fill_idx] = np.full(
                data.pos[i] - data.pos[i - 1] - 1, max(data.RD[i - 1], data.RD[i])
            )
            pos[fill_idx:fill_idx] = range(data.pos[i - 1] + 1, data.pos[i])
            cur_offset += data.pos[i] - data.pos[i - 1] - 1
        seg_idx[seg_id] = (start + sum_offset, end + sum_offset + cur_offset)
        sum_offset += cur_offset
    data.RD = np.asarray(rd)
    data.pos = np.asarray(pos)


def CBS_large_small_cluster(data: RDdata):
    seg_idx = CBS(data.RD)
    # fill_invalid_region(data, seg_idx)
    labels = np.empty(len(data.RD), dtype=int)
    for id, (start, end) in enumerate(seg_idx):
        labels[start:end] = id
    n_samples = len(data)

    size_clusters = np.bincount(labels)
    n_clusters = len(size_clusters)
    alpha = 0.7
    beta = 8
    sorted_cluster_indices = np.argsort(size_clusters * -1)  # reverse sort
    # Initialize the lists of index that fulfill the requirements by
    # either alpha or beta
    alpha_list = []
    beta_list = []
    for i in range(1, n_clusters):
        temp_sum = np.sum(size_clusters[sorted_cluster_indices[:i]])
        if temp_sum >= n_samples * alpha:
            alpha_list.append(i)
        if (
            size_clusters[sorted_cluster_indices[i - 1]]
            / size_clusters[sorted_cluster_indices[i]]
            >= beta
        ):
            beta_list.append(i)
    # Find the separation index fulfills both alpha and beta
    intersection = np.intersect1d(alpha_list, beta_list)
    if len(intersection) > 0:
        clustering_threshold = intersection[0]
    elif len(alpha_list) > 0:
        clustering_threshold = alpha_list[0]
    elif len(beta_list) > 0:
        clustering_threshold = beta_list[0]
    else:
        # cannot separate small cluster -> all normal
        return labels, None, None
    return (
        labels,
        sorted_cluster_indices[0:clustering_threshold],
        sorted_cluster_indices[clustering_threshold:],
    )


def LocalFeature(data: RDdata):
    labels, large_cluster_labels, small_cluster_labels = CBS_large_small_cluster(data)
    # denoise
    rd = data.RD
    # mean_rd, mad = estimate_location_and_mad(rd)
    # median_rd = np.median(rd)
    # mad = np.median(np.abs(rd - median_rd))
    # rd[rd > mean_rd + 800 * mad] = mean_rd + 800 * mad

    rd = denoise_tv_chambolle(rd.reshape(-1, 1), weight=0.1).ravel()

    # all normal
    if (
        large_cluster_labels is None
        or small_cluster_labels is None
        or len(small_cluster_labels) == 0
    ):
        return None, None, None

    n_clusters = len(large_cluster_labels) + len(small_cluster_labels)
    cluster_centers = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_centers[i] = np.mean(rd[np.where(labels == i)[0]])
    large_cluster_centers = cluster_centers[large_cluster_labels]

    # caculate local feature
    rd_data = rd.reshape(-1, 1)
    local_feature = np.zeros(len(data))
    small_indices = np.where(np.isin(labels, small_cluster_labels))[0]
    large_indices = np.where(np.isin(labels, large_cluster_labels))[0]

    # Find large cluster label which near to small cluster
    small_near = np.searchsorted(large_indices, small_indices)
    large_near_small_label_idx = np.empty(len(small_indices), dtype=int)
    locations = cluster_centers[labels]
    for i, idx in enumerate(small_indices):
        cur_center = cluster_centers[labels[idx]]
        tmp_dist = np.inf
        if small_near[i] < len(large_indices):
            right_label = labels[large_indices[small_near[i]]]
            right_center = cluster_centers[right_label]
            dist = abs(right_center - cur_center)
            if dist < tmp_dist:
                tmp_dist = dist
                locations[idx] = right_center
                large_near_small_label_idx[i] = np.where(
                    large_cluster_labels == right_label
                )[0][0]
        if small_near[i] > 0:
            left_label = labels[large_indices[small_near[i] - 1]]
            left_center = cluster_centers[left_label]
            dist = abs(left_center - cur_center)
            if dist < tmp_dist:
                tmp_dist = dist
                locations[idx] = left_center
                large_near_small_label_idx[i] = np.where(
                    large_cluster_labels == left_label
                )[0][0]

    # caculate distance
    dist_to_large_center = cdist(
        rd_data[small_indices, :], large_cluster_centers.reshape(-1, 1)
    )
    local_feature[small_indices] = dist_to_large_center[
        np.arange(len(small_indices)),
        large_near_small_label_idx,
    ]
    large_centers = cluster_centers[labels[large_indices]]
    local_feature[large_indices] = pairwise_distances_no_broadcast(
        rd_data[large_indices, :], large_centers.reshape(-1, 1)
    )

    return rd, local_feature, locations


def LMADCNV(data: RDdata, *, bp_per_bin: int = 1000) -> pd.DataFrame:
    rd, local_feature, locations = LocalFeature(data)
    if rd is None or local_feature is None or locations is None:
        return pd.DataFrame(columns=["start", "end", "type"])

    from pyod.models.mad import MAD

    clf = MAD()
    clf.fit(local_feature.reshape(-1, 1))

    from pythresh.thresholds import mad

    labels = mad.MAD().eval(clf.decision_scores_)
    labels[(rd < locations) & (labels == 1)] = -1

    return get_result(data, labels, bp_per_bin)

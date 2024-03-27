# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:50:55 2021

@author: karmeni1
"""

import json
import logging
import os
import sys

import numpy as np
import scipy.stats
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")

# make sure own modules are added to path
if "win" in sys.platform:
    home = os.environ["HOME"]
elif "linux" in sys.platform:
    home = os.environ["HOME"]
    scratch = os.environ["SCRATCH"]

# ===== CONSTANTS ===== #

# helper
class ProjectPaths:
    """
    helper class with constants holding frequently used paths
    """

    ROOT = os.path.join(home, "projects", "Pieman_ecog")
    DATA = os.path.join(home, ROOT, "data")
    RAW = os.path.join(home, DATA, "raw")
    OUTPUT = os.path.join(home, ROOT, "outputs")
    FIGURE = os.path.join(home, ROOT, "figures")

    if "linux" in sys.platform:
        DERIVED = os.path.join(scratch, "projects", "Pieman_ecog", "data", "derived")
    else:
        DERIVED = os.path.join(DATA, "derived")

    PIEMAN = os.path.join(home, RAW, "pieman")
    WAVFILE = os.path.join(PIEMAN, "Pieman_Original_MRI.wav")

    def __repr__(self):

        rep = (
            ".ROOT = {}\n".format(self.ROOT)
            + ".DATA = {}\n".format(self.DATA)
            + ".RAW = {}\n".format(self.RAW)
            + ".DERIVED {}\n".format(self.DERIVED)
            + ".PIEMAN {}\n".format(self.PIEMAN)
            + ".WAVFILE {}\n".format(self.WAVFILE)
        )

        return rep


# ===== MODEL PATHS ===== #

model_paths = {
    "glove-intact": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_glove_intact.npz"
    ),
    "gpt2-intact": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_1024-0-0_intact_10.npz"
    ),
    "gpt2-word": [
        os.path.join(
            ProjectPaths().DERIVED,
            "features",
            f"pieman_gpt2_words_scramble_{i:02d}.npz",
        )
        for i in range(10)
    ],  # scrambled randomizations
    "gpt2-sent": [
        os.path.join(
            ProjectPaths().DERIVED, "features", f"pieman_gpt2_sent_scramble_{i:02d}.npz"
        )
        for i in range(10)
    ],
    "gpt2-pars": [
        os.path.join(
            ProjectPaths().DERIVED, "features", f"pieman_gpt2_pars_scramble_{i:02d}.npz"
        )
        for i in range(10)
    ],
    "gpt2-intact-no-bigram": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_no-bigrams_1024_0_0.npz"
    ),
    "gpt2_1024_5_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_5_0.npz"
    ),  # context perturbations
    "gpt2_1024_50_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_50_0.npz"
    ),
    "gpt2_1024_100_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_100_0.npz"
    ),
    "gpt2_1024_200_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_200_0.npz"
    ),
    "gpt2_1024_0_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_0_0.npz"
    ),  # different context sizes
    "gpt2_512_0_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_512_0_0.npz"
    ),
    "gpt2_128_0_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_128_0_0.npz"
    ),
    "gpt2_64_0_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_64_0_0.npz"
    ),
    "gpt2_32_0_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_32_0_0.npz"
    ),
    "gpt2_5_0_0": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_5_0_0.npz"
    ),
    "gpt2_1024_50_5": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_50_5.npz"
    ),  # different amount of preserved tokens, 50 shuffled tokens
    "gpt2_1024_50_20": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_50_20.npz"
    ),
    "gpt2_1024_50_40": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_50_40.npz"
    ),
    "gpt2_1024_50_80": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_50_80.npz"
    ),
    "gpt2_1024_50_160": os.path.join(
        ProjectPaths().DERIVED, "features", "pieman_gpt2_intact_1024_50_160.npz"
    ),
}

# a dict that provides mapping between names on honeyserve storage
# and bids-like labels
sub_name2bids = {
    "bb05": "sub-005",
    "bb07": "sub-005",
    "scp18": "sub-008",
    "scp21": "sub-027",
}

ses_name2bids = {
    "pieman1": "001",
    "pieman2": "002",
}

# different recordings use different chan labels internally
# create a mapping form subject string to chan labels to track this

numstr = [str(i) for i in range(1, 65)]  # labels like "1", "2", "3"
gridstr = ["GRID{}".format(i) for i in range(1, 65)]

# these pieman sessions were apparently recorded with two different setups
# have a dict that tracks that
chan_name_labels = {
    "sub-004": gridstr,
    "sub-005": gridstr,
    "sub-007": gridstr,
    "sub-015": gridstr,
    "sub-016": gridstr,
    "sub-018": numstr,
    "sub-019": numstr,
    "sub-020": numstr,
    "sub-021": numstr,
}

# SUBJECT ID DICTS MAPPING BETWEEN DIFFERENT LABELS

fname = os.path.join(home, "projects", "Pieman_ecog", "bidsid2subid.json")
with open(fname, "r") as fh:
    bidsid2subid = json.load(fh)

fname = os.path.join(home, "projects", "Pieman_ecog", "subid2twh.json")
with open(fname, "r") as fh:
    subid2twh = json.load(fh)


# create a base class for storing subject data
class SubjData:
    def __init__(self, subject_id, sessions):

        # if input args are passed as ints, convert to BIDS-like
        # strings
        if type(subject_id) is int:
            subject_id = "sub-{:03d}".format(subject_id)
        if type(sessions[0]) is int:
            sessions = ["ses-{:03d}".format(ses) for ses in sessions]

        self.root = ProjectPaths().RAW
        self.audio = []
        self.name_bids = subject_id
        self.sessions = sessions
        self.cnt = []
        self.cnt_bids = []
        self.chan_labels = chan_name_labels[subject_id]
        self.twh = subid2twh[bidsid2subid[subject_id]]
        self.twh2 = bidsid2subid[subject_id]

        if subject_id in ["sub-016"]:
            elecnames_suffix = "_left.electrodeNames"
        elif subject_id in ["sub-015"]:
            elecnames_suffix = "_right.electrodeNames"
        else:
            elecnames_suffix = ".electrodeNames"

        self.electrodeNames = os.path.join(
            ProjectPaths().DATA,
            "freesurfer",
            self.twh,
            "elec_recon",
            self.twh + elecnames_suffix,
        )

        for ses in self.sessions:

            # bids_path = BIDSPath(subject=self.name_bids,
            #                    session=ses_name2bids[ses],
            #                    root=self.root)

            fname = "{}_{}_task-pieman.cnt".format(self.name_bids, ses)
            self.cnt.append(os.path.join(self.root, self.name_bids, fname))

            fname = f"{self.name_bids}_{ses}_audio.json"
            self.audio.append(
                os.path.join(ProjectPaths().DERIVED, self.name_bids, ses, fname)
            )

            fname = f"{self.name_bids}_{ses}_audio-xcorr.npy"
            self.audio.append(
                os.path.join(ProjectPaths().DERIVED, self.name_bids, ses, fname)
            )

            # contruct bids path too
            # self.cnt_bids.append(bids_path)

    def __repr__(self):

        rep = "SubjData({})".format(self.name_bids)

        return rep


class Dataset:
    def __init__(self):

        self.subs = None
        self.data = []

    def __repr__(self):

        rep = "Dataset() with fields:\n" + ".subs\n" + ".data\n"

        return rep

    def load_subinfo(self, subjects, sessions):

        self.subs = subjects

        for subid in subjects:

            logging.info("Loading info for {}".format(subid))

            # create instance of SubjData
            self.data.append(SubjData(sub_id=subid, sessions=sessions))


def load_ridge_outputs(subjects, session, feature):

    data = []

    for sub in tqdm(subjects, desc="file"):
        fname = f"{sub}_{session}_{feature}"
        fullfname = os.path.join(ProjectPaths().DERIVED, sub, session, "ridge", fname)
        data.append(dict(np.load(fullfname)))

    return data


def zscore(mat, return_unzvals=False):
    """Z-scores the rows of [mat] by subtracting off the mean and dividing
    by the standard deviation.
    If [return_unzvals] is True, a matrix will be returned that can be used
    to return the z-scored values to their original state.
    """
    zmat = np.empty(mat.shape, mat.dtype)
    unzvals = np.zeros((zmat.shape[0], 2), mat.dtype)
    for ri in range(mat.shape[0]):
        unzvals[ri, 0] = np.std(mat[ri, :])
        unzvals[ri, 1] = np.mean(mat[ri, :])
        zmat[ri, :] = (mat[ri, :] - unzvals[ri, 1]) / (1e-10 + unzvals[ri, 0])

    if return_unzvals:
        return zmat, unzvals

    return zmat


def center(mat, return_uncvals=False):
    """Centers the rows of [mat] by subtracting off the mean, but doesn't
    divide by the SD.
    Can be undone like zscore.
    """
    cmat = np.empty(mat.shape)
    uncvals = np.ones((mat.shape[0], 2))
    for ri in range(mat.shape[0]):
        uncvals[ri, 1] = np.mean(mat[ri, :])
        cmat[ri, :] = mat[ri, :] - uncvals[ri, 1]

    if return_uncvals:
        return cmat, uncvals

    return cmat


def unzscore(mat, unzvals):
    """Un-Z-scores the rows of [mat] by multiplying by unzvals[:,0] (the standard deviations)
    and then adding unzvals[:,1] (the row means).
    """
    unzmat = np.empty(mat.shape)
    for ri in range(mat.shape[0]):
        unzmat[ri, :] = mat[ri, :] * (1e-10 + unzvals[ri, 0]) + unzvals[ri, 1]
    return unzmat


def gaussianize(vec):
    """Uses a look-up table to force the values in [vec] to be gaussian."""
    ranks = np.argsort(np.argsort(vec))
    cranks = (ranks + 1).astype(float) / (ranks.max() + 2)
    vals = scipy.stats.norm.isf(1 - cranks)
    zvals = vals / vals.std()
    return zvals


def gaussianize_mat(mat):
    """Gaussianizes each column of [mat]."""
    gmat = np.empty(mat.shape)
    for ri in range(mat.shape[1]):
        gmat[:, ri] = gaussianize(mat[:, ri])
    return gmat


def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  ## negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)


def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.
    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array
    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))

    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d * mtx.T).T
    else:
        return d * mtx


import logging
import time


def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)

    for count, thing in enumerate(iterable):
        yield thing

        if not count % countevery:
            current_time = time.time()
            rate = float(count + 1) / (current_time - start_time)

            if rate > 1:  ## more than 1 item/second
                ratestr = "%0.2f items/second" % rate
            else:  ## less than 1 item/second
                ratestr = "%0.2f seconds/item" % (rate**-1)

            if total is not None:
                remitems = total - (count + 1)
                remtime = remitems / rate
                timestr = ", %s remaining" % time.strftime(
                    "%H:%M:%S", time.gmtime(remtime)
                )
                itemstr = "%d/%d" % (count + 1, total)
            else:
                timestr = ""
                itemstr = "%d" % (count + 1)

            formatted_str = "%s items complete (%s%s)" % (itemstr, ratestr, timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)


def nearest(array, value):
    """
    returns the index that corresponds to the nearest value in <array>
    """
    return np.abs(array - value).argmin()


def standardize_with_params(data, means, scales):
    """
    standardize data using means and scales
    """
    return (data - means) / scales


def transform(data, means, scales, pca_weights):
    """
    standardize and transform data
    """
    return (data - means) / scales @ pca_weights


def load_data(args):
    # broadband gamma
    fname_gamma = os.path.join(
        ProjectPaths().DERIVED,
        args.subject,
        args.session,
        f"{args.subject}_{args.session}_task-pieman_gamma.npz",
    )

    # audio data for envelope
    fname_audio = os.path.join(
        ProjectPaths().DERIVED,
        args.subject,
        args.session,
        f"{args.subject}_{args.session}_task-pieman_audio.npz",
    )

    # load repeat reliability scores for electrode selection
    fname_rr = os.path.join(
        ProjectPaths().DERIVED,
        args.subject,
        "ses-001",
        f"{args.subject}_repeat-reliability.npz",
    )

    loaded_vars = []
    for fname in (fname_gamma, fname_audio, fname_rr):

        logging.info(f"Loading {fname}...")
        loaded_vars.append(dict(np.load(fname, allow_pickle=True)))

    data, audio, rr = loaded_vars

    return data, audio, rr

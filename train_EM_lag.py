import argparse
import logging
import os

import numpy as np
import pandas as pd

from configs import train_EM_config as cfg
from LM import GPT2

# from sklearn.linear_model import RidgeCV, ridge_regression
from ridge import bootstrap_ridge, ridge
from utils import ProjectPaths, load_data, nearest

np.random.seed(42)


def main(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--gpt", type=str, required=True)

    if input_args:
        args = parser.parse_args(input_args)
    else:
        # input_args = ["--subject", "sub-007", "--session", "ses-001", "--gpt", "gpt2"]
        args = parser.parse_args()

    # load gpt
    gpt = GPT2(path=ProjectPaths().ROOT)

    # load stimulus data
    stim_data = pd.read_csv(
        os.path.join(
            ProjectPaths().PIEMAN,
            "annotations",
            "timings_and_annotations_without_punct_GPT2.tsv",
        ),
        sep="\t",
    )

    # load response data
    data, audio, rr = load_data(args)

    # get embeddings
    logging.info("Computing embeddings...")
    # get embeddings (standardized, pca-ed)
    embeddings, means, scales, pca_weights = gpt.get_embeddings(
        stim_data.token.tolist(), cfg.n_context_words, cfg.gpt2_layer, cfg.pca_dim
    )

    sel_chans = np.argsort(-rr["rr"])[
        : cfg.top_k
    ]  # select top k channels according to repeat reliability
    resp = data["data"][..., sel_chans, 0:-1].squeeze().T

    # get word/token onset times
    t_onsets = np.array(
        [nearest(data["times"], onset) for onset in stim_data.start.tolist()]
    )
    sample_rate = int(data["sfreq"])

    # get responses at different lags to word/token onset
    lags = cfg.lags
    responses_lag = np.zeros((len(t_onsets), resp.shape[1], lags.shape[0]))
    for i, onset in enumerate(t_onsets):
        for j, lag in enumerate(lags):
            responses_lag[i, :, j] = np.mean(
                resp[
                    (onset + cfg.window_from * sample_rate + lag * sample_rate)
                    .astype(int) : (
                        onset + cfg.window_to * sample_rate + lag * sample_rate
                    )
                    .astype(int),
                    :,
                ],
                axis=0,
            )
    responses = responses_lag.reshape((responses_lag.shape[0], -1))
    # zscore response
    responses = (responses - responses.mean(0)) / responses.std(0)

    # estimate encoding model with nfold cross-validation
    nchunks = int(np.ceil(responses.shape[0] / 5 / cfg.chunklen))
    # ridgecv = RidgeCV(alphas=cfg.alphas, alpha_per_target=True)
    # ridgecv.fit(embeddings, responses)
    # weights = ridgecv.coef_
    # alphas = ridgecv.alpha_
    # responses_pred = ridgecv.predict(embeddings)
    # corr = np.corrcoef(responses.T, responses_pred.T)
    # corr_score = np.diag(corr[: cfg.top_k * lags.shape[0], cfg.top_k * lags.shape[0] :])
    # # reshape correlation scores back to channels x lags
    # corr_score = corr_score.reshape((cfg.top_k, lags.shape[0]))
    weights, alphas, bscorrs = bootstrap_ridge(
        embeddings,
        responses,
        use_corr=True,
        alphas=cfg.alphas,
        nboots=cfg.nboots,
        chunklen=cfg.chunklen,
        nchunks=nchunks,
    )
    em_corr = np.zeros(cfg.top_k * lags.shape[0])
    for itarget in range(responses.shape[1]):
        alpha_idx = cfg.alphas == alphas[itarget]
        em_corr[itarget] = bscorrs[alpha_idx, itarget, :].mean()
    pass
    # generalization performance to other channels/lags
    gen_corr = np.zeros([cfg.top_k * lags.shape[0], cfg.top_k * lags.shape[0]])
    for i in range(responses.shape[1]):
        gen_pred = embeddings.dot(weights[:, i]).reshape(
            -1, 1
        )  # shape = (n_samples, 1)
        gen_corr[i, :] = np.corrcoef(gen_pred.T, responses.T)[0, 1:]

    # estimate noise model
    n_bootstrap_noise = 20
    n_samples = responses.shape[0]
    noise_model = np.zeros([cfg.top_k * lags.shape[0], cfg.top_k * lags.shape[0]])
    emscores = np.zeros([n_bootstrap_noise, cfg.top_k * lags.shape[0]])
    r_squared = np.zeros(cfg.top_k * lags.shape[0])
    sse = np.zeros(cfg.top_k * lags.shape[0])
    sst = np.zeros(cfg.top_k * lags.shape[0])
    for i in range(n_bootstrap_noise):
        permuted_idx = np.random.permutation(n_samples)
        heldout_idx = permuted_idx[: int(n_samples / 5)]
        train_idx = permuted_idx[int(n_samples / 5) :]
        tstim = embeddings[train_idx, :]
        tresp = responses[train_idx, :]
        hstim = embeddings[heldout_idx, :]
        hresp = responses[heldout_idx, :]
        # bs_weights = ridge_regression(tstim, tresp, alphas)
        # resids = hresp - hstim.dot(bs_weights.T)
        # bs_noise_model = resids.T.dot(resids)
        # corr = np.corrcoef(hresp.T, hstim.dot(bs_weights.T).T)
        bs_weights = ridge(tstim, tresp, alphas)
        resids = hresp - hstim.dot(bs_weights)
        r_squared += (
            1 - np.diag(resids.T.dot(resids)) / np.diag(hresp.T.dot(hresp))
        ) / n_bootstrap_noise
        sse += np.diag(resids.T.dot(resids)) / n_bootstrap_noise
        sst += np.diag(hresp.T.dot(hresp)) / n_bootstrap_noise
        bs_noise_model = resids.T.dot(resids)
        corr = np.corrcoef(hresp.T, hstim.dot(bs_weights).T)
        emscores[i] = np.diag(
            corr[: cfg.top_k * lags.shape[0], cfg.top_k * lags.shape[0] :]
        )
        noise_model += (
            bs_noise_model / np.diag(bs_noise_model).mean() / n_bootstrap_noise
        )
    emscores_sem = emscores.std(0) / np.sqrt(n_bootstrap_noise)
    emscores = emscores.mean(0)

    # save
    save_location = os.path.join(ProjectPaths.OUTPUT, args.subject)
    os.makedirs(save_location, exist_ok=True)
    np.savez(
        os.path.join(
            save_location,
            "encoding_model_lag_%s_%d_%d_%d"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        ),
        responses=responses,
        weights=weights,
        emscores=emscores,
        emscores_sem=emscores_sem,
        em_corr=em_corr,
        gen_corr=gen_corr,
        sse=sse,
        sst=sst,
        r_squared=r_squared,
        noise_model=noise_model,
        alphas=cfg.alphas,
        top_k=cfg.top_k,
        means=means,
        scales=scales,
        pca_weights=pca_weights,
        lags=lags,
    )

    # # plot encoding model performance as a function of lag
    # responses = np.zeros([len(t_onsets), resp.shape[1]])
    # for i, onset in enumerate(t_onsets):
    #     responses[i, :] = np.mean(resp[(onset + cfg.window_from * sample_rate).astype(int):(onset + cfg.window_to * sample_rate).astype(int), :], axis=0)

    # # zscore response
    # responses = (responses - responses.mean(0)) / responses.std(0)

    # # estimate encoding model
    # nchunks = int(np.ceil(responses.shape[0] / 5 / cfg.chunklen))
    # weights, alphas, bscorrs = bootstrap_ridge(embeddings, responses, use_corr = False, alphas = cfg.alphas,
    #     nboots = cfg.nboots, chunklen = cfg.chunklen, nchunks = nchunks)

    # # train model at word onset and evaluate at lags

    # lags = np.arange(-2, 2, 0.1)

    # n_bootstrap = 20
    # n_samples = responses.shape[0]
    # cv_scores = np.zeros([n_bootstrap, len(lags), cfg.top_k])
    # for ilag, lag in enumerate(lags):
    #     responses_lag = np.zeros([len(t_onsets), resp.shape[1]])
    #     for i, onset in enumerate(t_onsets):
    #         responses_lag[i, :] = np.mean(resp[(onset + cfg.window_from * sample_rate + lag * sample_rate).astype(int):(onset + cfg.window_to * sample_rate + lag * sample_rate).astype(int), :], axis=0)
    #         responses_lag = (responses_lag - responses_lag.mean(0)) / responses_lag.std(0)

    #     for iboot in range(n_bootstrap):
    #         permuted_idx = np.random.permutation(n_samples)
    #         heldout_idx = permuted_idx[:int(n_samples / 5)]
    #         train_idx = permuted_idx[int(n_samples / 5):]
    #         tstim = embeddings[train_idx, :]
    #         tresp = responses[train_idx, :]
    #         hstim = embeddings[heldout_idx, :]
    #         hresp = responses_lag[heldout_idx, :]
    #         # bs_weights = ridge(tstim, tresp, alphas)
    #         # for i_chan in range(cfg.top_k):
    #             # corr = np.corrcoef(hresp[:, i_chan], hstim.dot(bs_weights)[:, i_chan])
    #             # cv_scores[iboot, ilag, i_chan] = corr[0, 1]
    #         Rcmat = ridge_corr(tstim, hstim, tresp, hresp, alphas,
    #                                dtype=np.single, corrmin=0.2, singcutoff=1e-10, normalpha=False, use_corr=True)
    #         cv_scores[iboot, ilag, :] = Rcmat[iboot]

    # mean_cv_scores = cv_scores.mean(0)
    # std_cv_scores = cv_scores.std(0)

    # import matplotlib.pyplot as plt
    # # plt.plot(lags, bscorrs_lag.mean(1) - bscorrs_lag.std(1), color = 'blue')
    # # plt.plot(lags, bscorrs_lag.mean(1) + bscorrs_lag.std(1), color = 'blue')
    # for ic in range(5):
    #     plt.plot(lags, mean_cv_scores[:, -ic])
    #     plt.fill_between(lags, mean_cv_scores[:, -ic] - std_cv_scores[:, -ic] / np.sqrt(n_bootstrap), mean_cv_scores[:, -ic] + std_cv_scores[:, -ic] / np.sqrt(n_bootstrap), alpha = 0.2)
    # plt.xlabel("Time to word onset (s)")
    # plt.ylabel("Encoding model performance (Pearson's r)")
    # plt.xticks([-2, -1, 0, 1, 2])
    # # plt.yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    # plt.xlim([-2, 2])
    # # plt.ylim([0, 0.25])

    # plt.axvline(x = 0, color = 'black', linestyle = '--')
    # save_location = os.path.join(ProjectPaths.FIGURE, args.subject)
    # os.makedirs(save_location, exist_ok = True)
    # plt.savefig(os.path.join(save_location, 'encoding_model_performance_lag_gen.pdf'), dpi = 300)

    # # estimate encoding model
    # nchunks = int(np.ceil(responses.shape[0] / 5 / cfg.chunklen))
    # weights, alphas, bscorrs = bootstrap_ridge(embeddings, responses, use_corr = False, alphas = cfg.alphas,
    #     nboots = cfg.nboots, chunklen = cfg.chunklen, nchunks = nchunks)
    # bscorrs_mean = bscorrs.mean(1).mean(1)
    # i_alpha = np.argmax(bscorrs_mean, axis = 0)
    # bscorrs_lag[ilag, :, :] = bscorrs[i_alpha, :, :]


if __name__ == "__main__":
    main()

import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import train_EM_config as cfg
from utils import ProjectPaths


def main(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--gpt", type=str, required=True)

    if input_args:
        args = parser.parse_args(input_args)
    else:
        # input_args = ["--subject", "sub-004", "--session", "ses-001", "--gpt", "gpt2"]
        args = parser.parse_args()

    # create figure directory
    save_location = os.path.join(ProjectPaths().FIGURE, args.subject)
    os.makedirs(save_location, exist_ok=True)

    # load regression results
    results = np.load(
        os.path.join(
            ProjectPaths().OUTPUT,
            args.subject,
            "encoding_model_lag_%s_%d_%d_%d.npz"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        ),
        allow_pickle=True,
    )
    lags = results["lags"]
    # corr_score = results['corr_score']
    # mean_corr = results['mean_corr'].reshape((cfg.top_k, lags.shape[0]))
    emscores = results["emscores"].reshape((cfg.top_k, lags.shape[0]))
    emscores_sem = results["emscores_sem"].reshape((cfg.top_k, lags.shape[0]))
    em_corr = results["em_corr"].reshape((cfg.top_k, lags.shape[0]))
    gen_corr = results["gen_corr"]
    sse = results["sse"].reshape((cfg.top_k, lags.shape[0]))
    sst = results["sst"].reshape((cfg.top_k, lags.shape[0]))
    r_squared = results["r_squared"].reshape((cfg.top_k, lags.shape[0]))
    weights = results["weights"]
    noise_model = results["noise_model"]
    # precision = np.linalg.inv(noise_model * (1-cfg.shrinkage) + np.eye(noise_model.shape[0]) * cfg.shrinkage)
    precision = np.linalg.inv(noise_model)

    # plot encoding model performance as correlation between predicted and actual responses
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(15, 10))
    for i in range(4):
        for j in range(5):
            axs[i, j].fill_between(
                lags,
                emscores[i * 5 + j] - emscores_sem[i * 5 + j],
                emscores[i * 5 + j] + emscores_sem[i * 5 + j],
                alpha=0.5,
            )
            axs[i, j].plot(lags, emscores[i * 5 + j])
            axs[i, j].set_title("Channel %d" % (i * 5 + j))
            axs[i, j].set_xlim([lags[0], lags[-1]])
            axs[i, j].set_ylim([-0.1, 0.5])
            axs[i, j].axhline(0, color="grey", lw=0.5, linestyle="--")
            axs[i, j].axvline(0, color="grey", lw=0.5, linestyle="--")
            if i == 3:
                axs[i, j].set_xlabel("Lag (s)")
    fig.suptitle(args.subject + " Encoding model performance (Pearson's r)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            ProjectPaths().FIGURE,
            args.subject,
            "encoding_model_performance_lag_%s_%d_%d_%d.pdf"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        )
    )

    # plot encoding model performance as r squared
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(15, 10))
    for i in range(4):
        for j in range(5):
            axs[i, j].plot(lags, r_squared[i * 5 + j])
            axs[i, j].set_title("Channel %d" % (i * 5 + j))
            axs[i, j].set_xlim([lags[0], lags[-1]])
            axs[i, j].set_ylim([-0.1, 0.5])
            axs[i, j].axhline(0, color="grey", lw=0.5, linestyle="--")
            axs[i, j].axvline(0, color="grey", lw=0.5, linestyle="--")
            if i == 3:
                axs[i, j].set_xlabel("Lag (s)")
    fig.suptitle(args.subject + " Encoding model performance (R^2)")
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            ProjectPaths().FIGURE,
            args.subject,
            "encoding_model_r2_lag_%s_%d_%d_%d.pdf"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        )
    )

    # # plot sse
    # fig, axs = plt.subplots(nrows=4,ncols=5, figsize=(15,10))
    # for i in range(4):
    #     for j in range(5):
    #         axs[i,j].plot(lags, sse[i*5+j])
    #         axs[i,j].set_title('Channel %d' % (i*5+j))
    #         axs[i,j].set_xlim([lags[0], lags[-1]])
    #         # axs[i,j].set_ylim([-0.1, 0.5])
    #         axs[i,j].axhline(0, color='grey', lw=0.5, linestyle='--')
    #         axs[i,j].axvline(0, color='grey', lw=0.5, linestyle='--')
    #         if i == 3:
    #             axs[i,j].set_xlabel('Lag (s)')
    # fig.suptitle('Encoding model performance (SSE)')
    # fig.tight_layout()
    # plt.show()

    # # plot sst
    # fig, axs = plt.subplots(nrows=4,ncols=5, figsize=(15,10))
    # for i in range(4):
    #     for j in range(5):
    #         axs[i,j].plot(lags, sst[i*5+j])
    #         axs[i,j].set_title('Channel %d' % (i*5+j))
    #         axs[i,j].set_xlim([lags[0], lags[-1]])
    #         # axs[i,j].set_ylim([-0.1, 0.5])
    #         axs[i,j].axhline(0, color='grey', lw=0.5, linestyle='--')
    #         axs[i,j].axvline(0, color='grey', lw=0.5, linestyle='--')
    #         if i == 3:
    #             axs[i,j].set_xlabel('Lag (s)')
    # fig.suptitle('Encoding model performance (R^2)')
    # fig.tight_layout()
    # plt.show()

    # # plot covariance
    # fig, axs = plt.subplots(nrows=4,ncols=5)
    # emscores_error_inv = 1./np.diag(noise_model)
    # emscores_error_inv = emscores_error_inv.reshape((cfg.top_k, lags.shape[0]))
    # for i in range(4):
    #     for j in range(5):
    #         axs[i,j].plot(lags, emscores_error_inv[i*5+j])
    #         axs[i,j].set_title('Channel %d' % (i*5+j))
    #         axs[i,j].set_xlim([lags[0], lags[-1]])
    #         # axs[i,j].set_ylim([-0.1, 0.5])
    #         # axs[i,j].axhline(0, color='grey', lw=0.5, linestyle='--')
    # plt.show()

    # plot weight correlations
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.corrcoef(weights.T), cmap="coolwarm", vmax=1, vmin=-1)
    for i in range(cfg.top_k):
        rect = patches.Rectangle(
            (i * lags.shape[0], i * lags.shape[0]),
            lags.shape[0],
            lags.shape[0],
            linewidth=1,
            edgecolor="k",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_xticks([lags.shape[0] / 2, lags.shape[0] + lags.shape[0] / 2])
    ax.set_xticklabels(["Ch 0", "Ch 1"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(np.linspace(0, lags.shape[0], 3))
    ax.set_yticklabels(np.array([lags[0], 0, lags[-1]]).astype(int))
    ax.tick_params(axis="y")
    ax.set_xlabel("Channel x lag (s)")
    ax.set_ylabel("Channel x lag (s)")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    fig.colorbar(im, ax=ax, label="Pearson's r", shrink=0.5)
    fig.suptitle(args.subject + " Encoding model weight correlations")
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            ProjectPaths().FIGURE,
            args.subject,
            "encoding_model_weight_correlations_%s_%d_%d_%d.pdf"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        )
    )

    # plot generalization performance
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(gen_corr, cmap="coolwarm", vmax=0.5, vmin=-0.5)
    for i in range(cfg.top_k):
        rect = patches.Rectangle(
            (i * lags.shape[0], i * lags.shape[0]),
            lags.shape[0],
            lags.shape[0],
            linewidth=1,
            edgecolor="k",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_xticks([lags.shape[0] / 2, lags.shape[0] + lags.shape[0] / 2])
    ax.set_xticklabels(["Ch 0", "Ch 1"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(np.linspace(0, lags.shape[0], 3))
    ax.set_yticklabels(np.array([lags[0], 0, lags[-1]]).astype(int))
    ax.tick_params(axis="y")
    ax.set_xlabel("Channel x lag (s)")
    ax.set_ylabel("Channel x lag (s)")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    fig.colorbar(im, ax=ax, label="Pearson's r", shrink=0.5)
    fig.suptitle(args.subject + " Encoding model generalization performance")
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            ProjectPaths().FIGURE,
            args.subject,
            "encoding_model_generalization_performance_%s_%d_%d_%d.pdf"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        )
    )

    # plot noise model
    fig, ax = plt.subplots(figsize=(10, 8))
    max_val = np.max(np.abs(noise_model))
    im = ax.imshow(noise_model, cmap="coolwarm", vmax=max_val, vmin=-max_val)
    for i in range(cfg.top_k):
        rect = patches.Rectangle(
            (i * lags.shape[0], i * lags.shape[0]),
            lags.shape[0],
            lags.shape[0],
            linewidth=1,
            edgecolor="k",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.set_xticks([lags.shape[0] / 2, lags.shape[0] + lags.shape[0] / 2])
    ax.set_xticklabels(["Ch 0", "Ch 1"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(np.linspace(0, lags.shape[0], 3))
    ax.set_yticklabels(np.array([lags[0], 0, lags[-1]]).astype(int))
    ax.tick_params(axis="y")
    ax.set_xlabel("Channel x lag (s)")
    ax.set_ylabel("Channel x lag (s)")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    fig.colorbar(im, ax=ax, label="Covariance (a.u.)", shrink=0.5)
    fig.suptitle(args.subject + " Noise covariance matrix")
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            ProjectPaths().FIGURE,
            args.subject,
            "noise_covariance_matrix_%s_%d_%d_%d.pdf"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        )
    )

    # plot precision
    fig, ax = plt.subplots(figsize=(10, 8))
    max_val = np.max(np.abs(precision))
    im = ax.imshow(precision, cmap="coolwarm", vmax=max_val, vmin=-max_val)
    for i in range(cfg.top_k):
        rect = patches.Rectangle(
            (i * lags.shape[0], i * lags.shape[0]),
            lags.shape[0],
            lags.shape[0],
            linewidth=1,
            edgecolor="k",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_xticks([lags.shape[0] / 2, lags.shape[0] + lags.shape[0] / 2])
    ax.set_xticklabels(["Ch 0", "Ch 1"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(np.linspace(0, lags.shape[0], 3))
    ax.set_yticklabels(np.array([lags[0], 0, lags[-1]]).astype(int))
    ax.tick_params(axis="y")
    ax.set_xlabel("Channel x lag (s)")
    ax.set_ylabel("Channel x lag (s)")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    fig.colorbar(im, ax=ax, label="Contribution to decoder (a.u.)", shrink=0.5)
    fig.suptitle(
        args.subject
        + " Contributions to decoder (inverse of noise covariance matrix, channel x lag)"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            ProjectPaths().FIGURE,
            args.subject,
            "precision_matrix_%s_%d_%d_%d.pdf"
            % (args.gpt, cfg.gpt2_layer, cfg.n_context_words, cfg.pca_dim),
        )
    )


if __name__ == "__main__":
    main()

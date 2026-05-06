#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import os
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import torch.utils.data as Data

from vame.util.auxiliary import read_config
from vame.model.rnn_vae import RNN_VAE
from vame.model.dataloader import SEQUENCE_DATASET

use_gpu = torch.cuda.is_available()
if use_gpu:
    pass
else:
    torch.device("cpu")


def _safe_axes_2xn(axs, ncols: int):
    """确保 axs 形状为 (2, ncols)，避免 ncols=1 时 matplotlib 返回不规则数组。"""
    axs = np.array(axs)
    if ncols == 1:
        axs = axs.reshape(2, 1)
    else:
        axs = axs.reshape(2, ncols)
    return axs


def plot_reconstruction(
    filepath,
    test_loader,
    seq_len_half,
    model,
    model_name,
    FUTURE_DECODER,
    FUTURE_STEPS,
    suffix=None,
    n_examples=30,              # ★改为30
    save_overview=True,
    save_individual=True,
):
    # 取一个 batch
    dataiter = iter(test_loader)
    x = next(dataiter)
    x = x.permute(0, 2, 1)

    if use_gpu:
        data = x[:, :seq_len_half, :].type("torch.FloatTensor").cuda()
        data_fut = x[:, seq_len_half:seq_len_half + FUTURE_STEPS, :].type("torch.FloatTensor").cuda()
    else:
        data = x[:, :seq_len_half, :].type("torch.FloatTensor").to()
        data_fut = x[:, seq_len_half:seq_len_half + FUTURE_STEPS, :].type("torch.FloatTensor").to()

    if FUTURE_DECODER:
        x_tilde, future, latent, mu, logvar = model(data)

        fut_orig = data_fut.cpu().data.numpy()
        fut = future.cpu().detach().numpy()
    else:
        # 如果你的 config 里 prediction_decoder=False，那么不会生成“预测”子图
        x_tilde, latent, mu, logvar = model(data)

    data_orig = data.cpu().data.numpy()
    data_tilde = x_tilde.cpu().detach().numpy()

    # 你要的：30个样本（不超过batch大小）
    batch_n = data_orig.shape[0]
    n = int(n_examples)
    if n < 1:
        n = 1
    if n > batch_n:
        n = batch_n

    out_dir = os.path.join(filepath, "evaluate")
    os.makedirs(out_dir, exist_ok=True)

    # ======== FUTURE_DECODER：既画重建也画预测 ========
    if FUTURE_DECODER:
        title = "Reconstruction [top] and future prediction [bottom] of input sequence"

        # 1) 总览图：2 × n
        if save_overview:
            fig, axs = plt.subplots(2, n, figsize=(max(1, n) * 2.4, 6))
            fig.suptitle(title)
            axs = _safe_axes_2xn(axs, n)

            for i in range(n):
                axs[0, i].plot(data_orig[i, ...], color="k", label="Sequence Data")
                axs[0, i].plot(data_tilde[i, ...], color="r", linestyle="dashed", label="Sequence Reconstruction")

                axs[1, i].plot(fut_orig[i, ...], color="k")
                axs[1, i].plot(fut[i, ...], color="r", linestyle="dashed")

            axs[0, 0].set(xlabel="time steps", ylabel="reconstruction")
            axs[1, 0].set(xlabel="time steps", ylabel="predction")

            if suffix:
                out_name = f"Future_Reconstruction_overview_{suffix}.png"
            else:
                out_name = "Future_Reconstruction_overview.png"

            fig.savefig(os.path.join(out_dir, out_name), bbox_inches="tight")
            plt.close(fig)

        # 2) 单样本图：保存30张（每张2×1，方便你挑）
        if save_individual:
            for i in range(n):
                fig, axs = plt.subplots(2, 1, figsize=(3.6, 6))
                fig.suptitle(title)

                axs[0].plot(data_orig[i, ...], color="k", label="Sequence Data")
                axs[0].plot(data_tilde[i, ...], color="r", linestyle="dashed", label="Sequence Reconstruction")
                axs[0].set(xlabel="time steps", ylabel="reconstruction")

                axs[1].plot(fut_orig[i, ...], color="k")
                axs[1].plot(fut[i, ...], color="r", linestyle="dashed")
                axs[1].set(xlabel="time steps", ylabel="predction")

                if suffix:
                    out_name = f"Future_Reconstruction_sample_{i+1:02d}_{suffix}.png"
                else:
                    out_name = f"Future_Reconstruction_sample_{i+1:02d}.png"

                fig.savefig(os.path.join(out_dir, out_name), bbox_inches="tight")
                plt.close(fig)

    # ======== 非 FUTURE_DECODER：只画重建（保留兼容） ========
    else:
        if save_overview:
            fig, ax1 = plt.subplots(1, n, figsize=(max(1, n) * 2.4, 3.2))
            fig.suptitle("Reconstruction of input sequence")

            ax1 = np.array(ax1)
            if n == 1:
                ax1 = ax1.reshape(1)

            for i in range(n):
                ax1[i].plot(data_orig[i, ...], color="k", label="Sequence Data")
                ax1[i].plot(data_tilde[i, ...], color="r", linestyle="dashed", label="Sequence Reconstruction")

            fig.set_tight_layout(True)

            if suffix:
                out_name = f"Reconstruction_{model_name}_overview_{suffix}.png"
            else:
                out_name = f"Reconstruction_{model_name}_overview.png"

            fig.savefig(os.path.join(out_dir, out_name), bbox_inches="tight")
            plt.close(fig)

        if save_individual:
            for i in range(n):
                fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.2))
                fig.suptitle("Reconstruction of input sequence")

                ax.plot(data_orig[i, ...], color="k", label="Sequence Data")
                ax.plot(data_tilde[i, ...], color="r", linestyle="dashed", label="Sequence Reconstruction")

                if suffix:
                    out_name = f"Reconstruction_{model_name}_sample_{i+1:02d}_{suffix}.png"
                else:
                    out_name = f"Reconstruction_{model_name}_sample_{i+1:02d}.png"

                fig.savefig(os.path.join(out_dir, out_name), bbox_inches="tight")
                plt.close(fig)


def plot_loss(cfg, filepath, model_name):
    basepath = os.path.join(cfg["project_path"], "model", "model_losses")
    train_loss = np.load(os.path.join(basepath, "train_losses_" + model_name + ".npy"))
    test_loss = np.load(os.path.join(basepath, "test_losses_" + model_name + ".npy"))
    mse_loss_train = np.load(os.path.join(basepath, "mse_train_losses_" + model_name + ".npy"))
    mse_loss_test = np.load(os.path.join(basepath, "mse_test_losses_" + model_name + ".npy"))
    km_losses = np.load(os.path.join(basepath, "kmeans_losses_" + model_name + ".npy"))
    kl_loss = np.load(os.path.join(basepath, "kl_losses_" + model_name + ".npy"))
    fut_loss = np.load(os.path.join(basepath, "fut_losses_" + model_name + ".npy"))

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle("Losses of our Model")
    ax1.set(xlabel="Epochs", ylabel="loss [log-scale]")
    ax1.set_yscale("log")
    ax1.plot(train_loss, label="Train-Loss")
    ax1.plot(test_loss, label="Test-Loss")
    ax1.plot(mse_loss_train, label="MSE-Train-Loss")
    ax1.plot(mse_loss_test, label="MSE-Test-Loss")
    ax1.plot(km_losses, label="KMeans-Loss")
    ax1.plot(kl_loss, label="KL-Loss")
    ax1.plot(fut_loss, label="Prediction-Loss")
    ax1.legend()
    fig.savefig(os.path.join(filepath, "evaluate", "MSE-and-KL-Loss" + model_name + ".png"))
    plt.close(fig)


def eval_temporal(cfg, use_gpu, model_name, fixed, snapshot=None, suffix=None):
    SEED = 19
    ZDIMS = cfg["zdims"]
    FUTURE_DECODER = cfg["prediction_decoder"]
    TEMPORAL_WINDOW = cfg["time_window"] * 2
    FUTURE_STEPS = cfg["prediction_steps"]
    NUM_FEATURES = cfg["num_features"]
    if fixed is False:
        NUM_FEATURES = NUM_FEATURES - 2

    TEST_BATCH_SIZE = 64

    hidden_size_layer_1 = cfg["hidden_size_layer_1"]
    hidden_size_layer_2 = cfg["hidden_size_layer_2"]
    hidden_size_rec = cfg["hidden_size_rec"]
    hidden_size_pred = cfg["hidden_size_pred"]
    dropout_encoder = cfg["dropout_encoder"]
    dropout_rec = cfg["dropout_rec"]
    dropout_pred = cfg["dropout_pred"]
    softplus = cfg["softplus"]

    filepath = os.path.join(cfg["project_path"], "model")
    seq_len_half = int(TEMPORAL_WINDOW / 2)

    if use_gpu:
        torch.cuda.manual_seed(SEED)
        model = RNN_VAE(
            TEMPORAL_WINDOW,
            ZDIMS,
            NUM_FEATURES,
            FUTURE_DECODER,
            FUTURE_STEPS,
            hidden_size_layer_1,
            hidden_size_layer_2,
            hidden_size_rec,
            hidden_size_pred,
            dropout_encoder,
            dropout_rec,
            dropout_pred,
            softplus,
        ).cuda()
        model.load_state_dict(
            torch.load(os.path.join(cfg["project_path"], "model", "best_model", model_name + "_" + cfg["Project"] + ".pkl"))
        )
    else:
        model = RNN_VAE(
            TEMPORAL_WINDOW,
            ZDIMS,
            NUM_FEATURES,
            FUTURE_DECODER,
            FUTURE_STEPS,
            hidden_size_layer_1,
            hidden_size_layer_2,
            hidden_size_rec,
            hidden_size_pred,
            dropout_encoder,
            dropout_rec,
            dropout_pred,
            softplus,
        ).to()
        if not snapshot:
            model.load_state_dict(
                torch.load(
                    os.path.join(cfg["project_path"], "model", "best_model", model_name + "_" + cfg["Project"] + ".pkl"),
                    map_location=torch.device("cpu"),
                )
            )
        else:
            model.load_state_dict(torch.load(snapshot), map_location=torch.device("cpu"))

    model.eval()

    testset = SEQUENCE_DATASET(
        os.path.join(cfg["project_path"], "data", "train", ""),
        data="test_seq.npy",
        train=False,
        temporal_window=TEMPORAL_WINDOW,
    )
    test_loader = Data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)

    # ★固定生成30个：既重建也预测 + 保存总览 + 单张
    if not snapshot:
        plot_reconstruction(
            filepath,
            test_loader,
            seq_len_half,
            model,
            model_name,
            FUTURE_DECODER,
            FUTURE_STEPS,
            n_examples=30,          # ★改为30
            save_overview=True,
            save_individual=True,
        )
    else:
        plot_reconstruction(
            filepath,
            test_loader,
            seq_len_half,
            model,
            model_name,
            FUTURE_DECODER,
            FUTURE_STEPS,
            suffix=suffix,
            n_examples=30,          # ★改为30
            save_overview=True,
            save_individual=True,
        )

    plot_loss(cfg, filepath, model_name)


def evaluate_model(config, use_snapshots=False):
    """
    Evaluation of testset.

    Parameters
    ----------
    config : str
        Path to config file.
    use_snapshots : bool
        Whether to plot for all snapshots or only the best model.
    """
    config_file = Path(config).resolve()
    cfg = read_config(config_file)
    model_name = cfg["model_name"]
    fixed = cfg["egocentric_data"]

    if not os.path.exists(os.path.join(cfg["project_path"], "model", "evaluate")):
        os.mkdir(os.path.join(cfg["project_path"], "model", "evaluate"))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
        print("GPU active:", torch.cuda.is_available())
        print("GPU used:", torch.cuda.get_device_name(0))
    else:
        torch.device("cpu")
        print("CUDA is not working, or a GPU is not found; using CPU!")

    print("\n\nEvaluation of %s model. \n" % model_name)
    if not use_snapshots:
        eval_temporal(cfg, use_gpu, model_name, fixed)
    else:
        snapshots = os.listdir(os.path.join(cfg["project_path"], "model", "best_model", "snapshots"))
        for snap in snapshots:
            fullpath = os.path.join(cfg["project_path"], "model", "best_model", "snapshots", snap)
            epoch = snap.split("_")[-1]
            eval_temporal(cfg, use_gpu, model_name, fixed, snapshot=fullpath, suffix="snapshot" + str(epoch))

    print(
        "You can find the results of the evaluation in '/Your-VAME-Project-Apr30-2020/model/evaluate/' \n"
        "OPTIONS:\n"
        "- vame.pose_segmentation() to identify behavioral motifs.\n"
        "- re-run the model for further fine tuning. Check again with vame.evaluate_model()"
    )



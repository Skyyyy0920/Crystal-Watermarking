import omegaconf
import pytorch_lightning as pl
from tqdm import tqdm
from sklearn import metrics
from common.utils import PROJECT_ROOT
from cdvae.utils import *
from cdvae.pl_modules.model import CDVAE
from diffcsp.pl_modules.diffusion import CSPDiffusion


def run(cfg: DictConfig):
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    print(cfg)
    print(cfg.model)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Load checkpoint
    # ckpt = str(
    #     list(Path(r"G:\Crystal Watermarking\Crystal-Watermarking\cdvae\prop_models\cdvae\perovskite").glob('*.ckpt'))[0])
    # ckpt = str(list(Path(r"G:\Crystal Watermarking\Crystal-Watermarking\hydra\singlerun\2025-01-12\test").glob('*.ckpt'))[0])
    # ckpt = r"G:\Crystal Watermarking\Crystal-Watermarking\cdvae\prop_models\cdvae\perovskite\epoch=2664-step=61294.ckpt"
    # ckpt = str(
    #     list(Path(r"G:\Crystal Watermarking\Crystal-Watermarking\cdvae\prop_models\diffcsp\perovskite").glob('*.ckpt'))[
    #         0])
    ckpt = r"G:\Crystal Watermarking\Crystal-Watermarking\cdvae\prop_models\diffcsp\perovskite\epoch=124-step=1499.ckpt"

    print(ckpt)
    name = 'cspdiff'
    # name = 'CDVAE'
    if name == 'CDVAE':
        model = CDVAE.load_from_checkpoint(ckpt)
    else:
        # model = CSPDiffusion.load_from_checkpoint(ckpt)
        model = model.load_from_checkpoint(ckpt)

    model.eval()
    print(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Setup data
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()[0]

    results = []
    no_w_metrics = []
    w_metrics = []

    def get_watermarking_pattern(batch_size, pattern_size=3):
        # 为9维特征设计水印模式
        positions = np.random.choice(9, size=pattern_size, replace=False)
        amplitudes = torch.rand(batch_size, pattern_size) * 0.1 - 0.05  # [-0.05, 0.05]
        return positions, amplitudes

    def detect_watermark(features, positions, amplitudes):
        # features: [batch_size, feature_dim]
        extracted = features[:, positions]
        similarity = F.cosine_similarity(extracted, amplitudes, dim=1)
        return similarity.mean() > 0.7, similarity.mean()

    for i, batch in enumerate(tqdm(test_loader)):
        batch = batch.to(device)
        seed = i + 42
        # Generation without watermark
        set_random_seed(seed)

        # 生成初始噪声
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords
        rand_l = torch.randn_like(lattices)  # [batch_size, 3, 3]
        rand_x = torch.randn_like(frac_coords)

        # 将噪声展平用于水印操作
        z_flat = rand_l.view(rand_l.size(0), -1)  # [batch_size, 9]

        # 生成带水印的噪声
        watermarked_flat = z_flat.clone()
        positions, amplitudes = get_watermarking_pattern(watermarked_flat.shape[0])
        watermarked_flat[:, positions] += amplitudes
        watermarked_rand_l = watermarked_flat.view_as(rand_l)

        # 生成样本（这里需要您的扩散生成函数）
        # 未加水印的生成
        clean_sample = model.decode_stats(
            rand_l, rand_x, batch.num_atoms, batch.lengths, batch.angles
        )

        # 加水印的生成
        watermarked_sample = model.decode_stats(
            watermarked_rand_l, rand_x, batch.num_atoms, batch.lengths, batch.angles
        )

        # 检测水印（需要从生成的样本中提取特征）
        # 这里假设我们可以从生成的晶格参数中提取特征
        generated_lattice = watermarked_sample.lattice_matrix
        detection_feature = generated_lattice.view(generated_lattice.size(0), -1)
        is_watermarked, similarity = detect_watermark(detection_feature, positions, amplitudes)
        print(f"Watermark detected: {is_watermarked}")
        print(f"Similarity score: {similarity:.4f}")

        # 检测未加水印样本
        clean_lattice = clean_sample.lattice_matrix
        clean_feature = clean_lattice.view(clean_lattice.size(0), -1)
        is_watermarked, similarity = detect_watermark(clean_feature, positions, amplitudes)
        print(f"Clean data watermark detected: {is_watermarked}")
        print(f"Clean data similarity score: {similarity:.4f}")



    # for i, batch in enumerate(tqdm(test_loader)):
    #     batch = batch.to(device)
    #     seed = i + 42
    #
    #     # Generation without watermark
    #     set_random_seed(seed)
    #     mu, log_var, z_no_w = model.encode(batch)  # [batch_size, 256]
    #     outputs_no_w = model.decode_stats(z_no_w, batch.num_atoms)
    #
    #     # Generation with watermark
    #     z_w = copy.deepcopy(z_no_w)
    #
    #     # 获取水印模式
    #     positions, amplitudes = get_watermarking_pattern(z_w.shape[0])
    #
    #     # 添加水印
    #     watermarked = add_watermark(z_w, positions, amplitudes)
    #
    #     outputs_no_w = model.decode_stats(z_no_w, batch.num_atoms, batch.lengths, batch.angles)
    #
    #     # 检测水印
    #     is_watermarked, similarity = detect_watermark(watermarked, positions, amplitudes)
    #     print(f"Watermark detected: {is_watermarked}")
    #     print(f"Similarity score: {similarity:.4f}")
    #
    #     # 测试未水印的数据
    #     is_watermarked, similarity = detect_watermark(z_no_w, positions, amplitudes)
    #     print(f"Clean data watermark detected: {is_watermarked}")
    #     print(f"Clean data similarity score: {similarity:.4f}")
    #
    #     #     results.append({
    #     #         'no_w_metric': no_w_metric,
    #     #         'w_metric': w_metric,
    #     #     })
    #     #
    #     #     no_w_metrics.append(-no_w_metric)
    #     #     w_metrics.append(-w_metric)
    #     #
    #     # # Calculate ROC and metrics
    #     # preds = no_w_metrics + w_metrics
    #     # t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
    #     #
    #     # fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    #     # auc = metrics.auc(fpr, tpr)
    #     # acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    #     # low = tpr[np.where(fpr < .01)[0][-1]]
    #     #
    #     # print(f'AUC: {auc}, ACC: {acc}, TPR@1%FPR: {low}')


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()

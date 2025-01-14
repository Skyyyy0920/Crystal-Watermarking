import omegaconf
import pytorch_lightning as pl
from tqdm import tqdm
from sklearn import metrics
from common.utils import PROJECT_ROOT
from cdvae.utils import *
from pl_modules.model import CDVAE


def run(cfg: DictConfig):
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    # hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    # model: pl.LightningModule = hydra.utils.instantiate(
    #     cfg.model,
    #     optim=cfg.optim,
    #     data=cfg.data,
    #     logging=cfg.logging,
    #     _recursive_=False,
    # )

    # Load checkpoint
    ckpt = str(
        list(Path(r"W:\Crystal Watermarking\Crystal-Watermarking\cdvae\prop_models\perovskite").glob('*.ckpt'))[0])
    ckpt = str(list(Path(r"W:\Crystal Watermarking\cdvae-main\hydra\singlerun\2025-01-12\test").glob('*.ckpt'))[0])
    print(ckpt)
    name = 'CDVAE'
    if name == 'CDVAE':
        model = CDVAE.load_from_checkpoint(ckpt)
    else:
        model = None

    model.eval()
    print(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Initialize watermark pattern
    mu, log_var, z_no_w = model.encode(batch)
    gt_patch = get_watermarking_pattern(z_no_w, cfg.watermark, device)

    results = []
    no_w_metrics = []
    w_metrics = []

    # Setup data TODO
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    for i, batch in enumerate(tqdm(test_loader)):
        batch = batch.to(device)
        seed = i + cfg.watermark.gen_seed

        # Generation without watermark
        set_random_seed(seed)
        mu, log_var, z_no_w = model.encode(batch)  # [batch_size, 256]
        outputs_no_w = model.decode_stats(z_no_w, batch.num_atoms)

        # Generation with watermark
        z_w = copy.deepcopy(z_no_w)

        # Get watermark mask and inject
        watermarking_mask = get_watermarking_mask(z_w, cfg.watermark, device)
        z_w = inject_watermark(z_w, watermarking_mask, gt_patch, cfg.watermark)
        outputs_w = model.decode_stats(z_w, batch.num_atoms)

        # Evaluate structures
        no_w_metric, w_metric = eval_watermark(outputs_no_w, outputs_w, watermarking_mask, gt_patch, cfg.watermark)

        results.append({
            'no_w_metric': no_w_metric,
            'w_metric': w_metric,
        })

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

    # Calculate ROC and metrics
    preds = no_w_metrics + w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < .01)[0][-1]]

    print(f'AUC: {auc}, ACC: {acc}, TPR@1%FPR: {low}')


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()

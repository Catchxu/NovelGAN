from benchmark.run_benchmark import cell_detect_bench

data_cfg = {
    'PBMC': {
        'train': '/volume1/home/kxu/scdata/PBMC_xsun/PBMC1.h5ad',
        'test1': '/volume1/home/kxu/scdata/PBMC_xsun/PBMC2.h5ad',
        'test2': '/volume1/home/kxu/scdata/PBMC_xsun/PBMC3.h5ad'
    }
}

method_cfg = {'NovelGAN': 1}
metrics = ['ARI', 'NMI']

train, test1, test2 = cell_detect_bench(
    data_cfg, method_cfg, metrics, log_path='loguru.log', random_state=100
)
# rm_cache("./cache")

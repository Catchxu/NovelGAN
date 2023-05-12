from benchmark.run_benchmark import cell_detect_bench

data_cfg = {
    'pbmcA+HL': {
        'data1_path': './scdata/pbmcA/',
        'data2_path': './scdata/HL/',
    }
}

method_cfg = {'NovelGan': 1}
scales = ['small']
metrics = ['ARI', 'NMI']

cell_detect_bench(
    data_cfg, method_cfg, scales, metrics, log_path='loguru.log', random_state=100
)
# rm_cache("./cache")
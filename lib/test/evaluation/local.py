from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/got10k_lmdb'
    settings.got10k_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/itb'
    settings.lasot_extension_subset_path_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/lasot_lmdb'
    settings.lasot_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/lasot'
    settings.network_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/nfs'
    settings.otb_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/otb'
    settings.prj_dir = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net'
    settings.result_plot_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/output/test/result_plots'
    settings.results_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/output'
    settings.segmentation_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/output/test/segmentation_results'
    settings.tc128_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/trackingnet'
    settings.uav_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/uav'
    settings.vot18_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/vot2018'
    settings.vot22_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/vot2022'
    settings.vot_path = '/data/lizf/ijcai2024/base_whispers2023/0317/3dConv/submit_guthub/tmo/DaSSP-Net/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


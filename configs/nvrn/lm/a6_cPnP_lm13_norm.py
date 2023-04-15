_base_ = ["../../_base_/gdrn_base.py"]

# OUTPUT_DIR = "debug/gdrn/lm/a6_cPnP_lm13_norm"
# OUTPUT_DIR = "output/gdrn/lm/a6_cPnP_lm13_norm_59"
# OUTPUT_DIR = "output/gdrn/lm/a6_cPnP_lm13_norm_320"
OUTPUT_DIR = "output/nvrn/lm/a6_cPnP_lm13_norm_320v42"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),"
        # 0"Sometimes(0.5, Affine(scale=(1.0, 1.2))),"
        "Sometimes(0.5, GaussianBlur(np.random.rand())),"
        "Sometimes(0.5, Add((-20, 20), per_channel=0.3)),"
        "Sometimes(0.4, Invert(0.20, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),"
        "Sometimes(0.5, Multiply((0.7, 1.4))),"
        "Sometimes(0.5, ContrastNormalization((0.5, 2.0), per_channel=0.3))"
        "], random_order=False)"
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=64,
    TOTAL_EPOCHS=400, # 160
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.6,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=5e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    MAX_TO_KEEP=1,
    CHECKPOINT_PERIOD=20,
)

DATASETS = dict(
    TRAIN=("lm_13_train", "lm_imgn_13_train_1k_per_obj"),
    TEST=("lm_13_test",),
    DET_FILES_TEST=("datasets/BOP_DATASETS/lm/test/test_bboxes/bbox_faster_all.json",),
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
        ROT_HEAD=dict(
            FREEZE=False,
            ROT_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            DEPTH_CLASS_AWARE=False,
            NORM_LW=3.0,
            REGION_CLASS_AWARE=False,
            NUM_REGIONS=64,
            # Cross channel loss
            CROSS_NORM_LOSS_TYPE="L1",
            CROSS_NORM_LW=1.0,
            CROSS_NORM_LOSS_START=0.3,
            DEPTH_LW=1.0,
            DEPTH2NORM_LOSS_START=0.4,
            DEPTH2NORM_LW=0.5,
        ),
        PNP_NET=dict(
            R_ONLY=False,
            T_ONLY=True,
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            WITH_DEPTH=False,
            WITH_MASK=False,
            COORD_TYPE='abs',
            ROT_TYPE="allo_rot6d",
            # TRANS_TYPE="centroid_z_rec",
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=True,
            PM_LW=0.0,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=2.0,
            Z_LOSS_TYPE="L1",
            Z_LW=2.0,
            # Cross layer loss
            CROSS_R_LOSS_TYPE="L1",
            CROSS_R_LW=0,
            CROSS_R_LOSS_START=1.0,
        ),
        SVD_NET=dict(
            ENABLE=True,
            REGION_ATTENTION=True,
            WITH_2D_COORD=False,
            WITH_DEPTH=False,
            WITH_MASK=False,
            ROT_TYPE="allo_rot6d",
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=True,
            PM_LW=1.0,
            # Cross layer loss
            CROSS_R_LOSS_TYPE="L1",
            CROSS_R_LW=1.0,
            CROSS_R_LOSS_START=0.4,
        ),
        TRANS_HEAD=dict(FREEZE=True),
    ),
)

TEST = dict(
    EVAL_PERIOD=0, 
    VIS=False, 
    TEST_BBOX_TYPE="est",
    USE_SVD=True
)  # gt | est

# DATALOADER = dict(
#     # Number of data loading threads
#     NUM_WORKERS=8,
# )
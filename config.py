class Config:
    VERBOSE = False

    # TRAINING SETTINGS
    MODEL_NAME = "resnet18"
    OPTIMIZER = "Adam"
    LR = 0.001
    EPOCHS = 8
    DO_AUGMENTATION = False
    SCHEDULER = True
    PRETRAINED = True
    FINE_TUNING = True
    INPUT_SIZE = 224
    BATCH_SIZE = 13
    GPU = True
    EARLY_STOPPING_PATIENCE = 5

    # AUGMENTATION / PREPROCESSING
    ROTATION_DEGREES = 45
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]

    # INFERENCE / SCORING
    INFERENCE_BATCH = 50
    ALLOWED_EXTS = [".png", ".jpg", ".jpeg"]

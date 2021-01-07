from models.wideresnet import WideResNet


def create_model(use_cuda, n_classes, ema=False):
    model = WideResNet(num_classes=n_classes)
    model = model.cuda() if use_cuda else model

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def setup_models(use_cuda, n_classes):
    print("==> creating WRN-28-2")
    model = create_model(use_cuda, n_classes)
    ema_model = create_model(use_cuda, n_classes, ema=True)
    return model, ema_model

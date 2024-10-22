from config import DataArgs, ModelArgs, TrainArgs, InferenceArgs


def get_args():
    """ parse all config args into one place """

    data_args = DataArgs()
    model_args = ModelArgs()
    train_args = TrainArgs()
    inference_args = InferenceArgs()

    args = {}

    for config_args in [model_args, train_args, data_args, inference_args]:
        args.update(vars(config_args))

    return args

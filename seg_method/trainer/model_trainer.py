import json
class SimpleTrainer(object):
    def __init__(self, config_path):
        train_config = json.load(open(config_path))
        self.model_name = train_config['model_config']['model_name']
        print(self.model_name)
        if self.model_name == "dual_MBConv_VAE":
            pass
        elif self.model_name == "monai_3D_unet":
            pass
    def train(self):
        pass
    def evaluate(self):
        pass




Trainer = SimpleTrainer(config_path="../../train_configuration/test.json")
from seg_method.trainer import model_trainer
if __name__ == '__main__':
    Trainer = model_trainer.SimpleTrainer(config_path='../train_configuration/test_swinunter_monai.json')
    Trainer.train()

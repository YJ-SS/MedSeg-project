from seg_method.trainer import model_trainer
if __name__ == '__main__':
    Trainer = model_trainer.SimpleTrainer(config_path='../train_configuration/dual_mbconv_vae_carnial_seg.json')
    # Train model1
    Trainer.train()
    # Evaluate model1
    # Trainer.evaluate()



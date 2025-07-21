from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'cityscapes'
        self.data_root = './dataset/my_cityscape'
        self.num_class = 2

        # Model
        self.model = 'fastscnn'
        # self.encoder = 'resnet101'
        # self.decoder = 'deeplabv3p'

        # Training
        self.total_epoch = 20
        self.train_bs = 2
        self.loss_type = 'ohem'
        self.optimizer_type = 'adam'
        self.logger_name = 'seg_trainer'
        self.use_aux = False
        self.resume_training = False


        # Validating
        self.val_bs = 1

        # Testing
        self.is_testing = True
        self.test_bs = 2
        self.test_data_folder = './test_data'
        self.load_ckpt_path = './save/best_fastscnn.pth'
        self.save_mask = True
        self.load_ckpt = True

        # Training setting
        self.use_ema = False

        # Augmentation
        self.crop_size = 768
        self.randscale = [-0.5, 1.0]
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = '/path/to/your/teacher/checkpoint'
        self.teacher_model = 'fastscnn'
        self.teacher_encoder = 'resnet101'
        self.teacher_decoder = 'deeplabv3p'

from torch import optim

from models.model_interaction import InteractionModel
from models.model_sampling import SamplingModel
from TorchDockingFFT import TorchDockingFFT
from TrainerT1 import TrainerT1
from TrainerT0 import TrainerT0
from UtilityFunctions import UtilityFunctions


class TrainerWrapperT0:
    """
    Wrapper to train variations on the core model
    """

    def __init__(self,
                 device,
                 dtype,
                 zero_feature=False,
                 docked_complex=None,
                 experiment=None,
                 learning_rate=1e-3,
                 train_model=False,
                 resume_training=False,
                 resume_epoch=0,
                 train_epochs=0,
                 evaluate_model=False,
                 resolution_in_angstroms=2.0,
                 box_dim=50,
                 padded_dim=100,
                 eval_freq=10,
                 so3_angles=None,
                 CD=False,
                 NS=False,
                 JT=False,
                 target_checking=False,
                 trainer_debug=False,
                 fft_debug=False,
                 ):

        self.docked_complex = docked_complex
        self.experiment = experiment
        self.learning_rate = learning_rate
        self.train_model = train_model
        self.resume_training = resume_training
        self.resume_epoch = resume_epoch
        self.train_epochs = train_epochs
        self.evaluate_model = evaluate_model
        self.eval_freq = eval_freq
        self.trainer_debug = trainer_debug
        self.fft_debug = fft_debug
        self.device = device
        self.dtype = dtype

        ## initialize FFT docking algorithm to score the correlation (Energy, not free energy until LogSumExp(Energy))
        DockingFFT = TorchDockingFFT(box_dim=box_dim,
                                     padded_dim=padded_dim,
                                     # training=self.train_model,
                                     docked_complex=docked_complex,
                                     zero_feature=zero_feature,
                                     so3_angles=so3_angles,
                                     debug=fft_debug, device=self.device, dtype=self.dtype)

        ModelSampler = SamplingModel(DockingFFT, docked_complex=docked_complex, zero_feature=zero_feature, device=self.device, dtype=self.dtype, training=self.train_model)

        ModelSampler.apply(UtilityFunctions(device=self.device, dtype=self.dtype).weights_init)
        optimizer = optim.Adam(ModelSampler.parameters(), lr=learning_rate)

        self.Trainer = TrainerT0(
            device=self.device,
            dtype=self.dtype,
            docked_complex=docked_complex,
            experiment=experiment,
            dockingFFT=DockingFFT,
            sampling_model=ModelSampler,
            model_optimizer=optimizer,
            resolution_in_angstroms=resolution_in_angstroms,
            eval_freq=self.eval_freq,
            target_checking=target_checking,
            so3_angles=so3_angles,
            debug=trainer_debug,
        )

    def call_trainer(self,
                     train_stream,
                     valid_stream,
                     test_stream):
        """
        :param self:
        """
        if self.train_model:
            self.Trainer.run_trainer(
                train_epochs=self.train_epochs,
                train_stream=train_stream,
                valid_stream=valid_stream,
                test_stream=test_stream)

        if self.evaluate_model:
            self.Trainer.run_trainer(
                resume_training=True, 
                resume_epoch=self.resume_epoch,
                train_epochs=self.train_epochs,
                train_stream=None,
                valid_stream=valid_stream,
                test_stream=test_stream,
            )

        if self.resume_training:
            self.Trainer.run_trainer(
                resume_training=self.resume_training,
                resume_epoch=self.resume_epoch,
                train_epochs=self.train_epochs,
                train_stream=train_stream,
                valid_stream=valid_stream,
                test_stream=test_stream,
            )


class TrainerWrapperT1:
    """
    Wrapper to train variations on the core model
    """

    def __init__(self,
                 device,
                 dtype,
                 zero_feature=False,
                 docked_complex=None,
                 experiment=None,
                 learning_rate=1e-3,
                 interaction_learning_rate=1e-1,
                 training_case='scratch',
                 path_pretrain=None,
                 train_model=False,
                 resume_training=False,
                 resume_epoch=0,
                 train_epochs=0,
                 evaluate_model=False,
                 resolution_in_angstroms=2.0,
                 box_dim=50,
                 padded_dim=100,
                 eval_freq=10,
                 so3_angles=None,
                 CD=False,
                 NS=False,
                 JT=False,
                 target_checking=False,
                 trainer_debug=False,
                 fft_debug=False,
                 inference=False,
                 ):

        
        self.docked_complex = docked_complex
        self.zero_feature = zero_feature
        self.experiment = experiment
        self.learning_rate = learning_rate
        self.interaction_learning_rate = interaction_learning_rate
        self.train_model = train_model
        self.resume_training = resume_training
        self.resume_epoch = resume_epoch
        self.train_epochs = train_epochs
        self.evaluate_model = evaluate_model
        self.eval_freq = eval_freq
        self.trainer_debug = trainer_debug
        self.fft_debug = fft_debug
        self.device = device
        self.dtype = dtype

        ## initialize FFT docking algorithm to score the correlation (Energy, not free energy until LogSumExp(Energy))
        DockingFFT = TorchDockingFFT(box_dim=box_dim,
                                     padded_dim=padded_dim,
                                     # training=self.train_model,
                                     docked_complex=docked_complex,
                                     zero_feature=zero_feature,
                                     so3_angles=so3_angles,
                                     debug=fft_debug, device=self.device, dtype=self.dtype)

        ModelSampler = SamplingModel(DockingFFT, docked_complex=docked_complex, zero_feature=zero_feature, device=self.device, dtype=self.dtype, training=self.train_model)

        ModelSampler.apply(UtilityFunctions(device=self.device, dtype=self.dtype).weights_init)
        optimizer = optim.Adam(ModelSampler.parameters(), lr=learning_rate)

        interaction_model = InteractionModel(device=self.device, dtype=self.dtype).to(device=self.device, dtype=self.dtype)
        interaction_optimizer = optim.Adam(interaction_model.parameters(), lr=interaction_learning_rate)
        # print("wrapper interaction_learning_rate",interaction_learning_rate)

        self.Trainer = TrainerT1(
            device=self.device,
            dtype=self.dtype,
            docked_complex=docked_complex,
            zero_feature=zero_feature,
            experiment=experiment,
            dockingFFT=DockingFFT,
            sampling_model=ModelSampler,
            model_optimizer=optimizer,
            interaction_model=interaction_model,
            interaction_optimizer=interaction_optimizer,
            training_case=training_case,
            path_pretrain=path_pretrain,
            resolution_in_angstroms=resolution_in_angstroms,
            eval_freq=self.eval_freq,
            target_checking=target_checking,
            so3_angles=so3_angles,
            debug=trainer_debug,
            inference=inference,
        )

    def call_trainer(self,
                     train_stream,
                     valid_stream,
                     test_stream):
        """
        :param self:
        """
        if self.train_model:
            self.Trainer.run_trainer(
                train_epochs=self.train_epochs,
                train_stream=train_stream,
                valid_stream=valid_stream,
                test_stream=test_stream)

        if self.evaluate_model:
            self.Trainer.run_trainer(
                resume_training=True, 
                resume_epoch=self.resume_epoch,
                train_epochs=self.train_epochs,
                train_stream=None,
                valid_stream=valid_stream,
                test_stream=test_stream,
            )

        if self.resume_training:
            self.Trainer.run_trainer(
                resume_training=self.resume_training,
                resume_epoch=self.resume_epoch,
                train_epochs=self.train_epochs,
                train_stream=train_stream,
                valid_stream=valid_stream,
                test_stream=test_stream,
            )

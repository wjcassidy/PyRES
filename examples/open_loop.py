# ==================================================================
# ============================ IMPORTS =============================
import argparse
import time
import sys
import os

import flamo.functional
from flamo.functional import mag2db, db2mag

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# PyTorch
import torch
# FLAMO
from flamo import system, dsp
from flamo.optimize.dataset import Dataset, load_dataset
from flamo.optimize.trainer import Trainer
# PyRES
from PyRES.res import RES
from PyRES.physical_room import PhRoom_dataset
from PyRES.virtual_room import random_FIRs
from PyRES.loss_functions import MSE_evs_mod
from PyRES.functional import system_equalization_curve
from PyRES.plots import plot_evs_compare, plot_spectrograms_compare, plot_irs_compare
from PyRES.virtual_room import VrRoom

###########################################################################################
# In this example, we train a virtual room to equalize the RES.
# The physical room is simulated with the PhRoom_wgn class.
# The virtual room is a mixing matrix of finite-impulse-response (FIR) filters.
# For more information about the classes Dataset and Trainer, please refer to the FLAMO
# documentation.
# The training pipeline is as follows:
# 1. Initialize the physical room and the virtual room.
# 2. Initialize the RES with the physical and virtual rooms.
# 3. Define the model as the open loop of the RES.
# 4. Initialize the dataset with the input and target signals.
#    The input signal is a batch of unit impulses.
#    The target signal is the equalization curve of the RES.
# 5. Initialize the trainer with the model and the dataset.
# 6. Define the loss function as the mean squared error between the target and the eigenvalues
#    of the RES open loop.
# 7. Train the model with the trainer.
# 8. Plot the eigenvalues and the spectrograms to compare the system responses before and
#    after optimization.
# 9. Save the model parameters (optional).
# This example, with the current training parameters, is meant as a proof of concept.
# By increasing the size of the dataset (see hyperparameter `--num`), the optimizer will
# iterate over more training examples (more unit impulses), and the result will improve.

# Reference:
#     De Bortoli, G., Dal Santo, G., Prawda, K., Lokki, T., Välimäki, V., and Schlecht, S. J.
#     "Differentiable Active Acoustics: Optimizing Stability via Gradient Descent"
#     Proceedings of the International Conference on Digital Audio Effects, pp. 254-261, 2024.
###########################################################################################

torch.manual_seed(141122)


class VrRoom_FIRS_WGN(VrRoom):
    f"""
    This assumes you have used a full matrix of FIR filters and that you want to use one WGN reverb ir per each loudspeaker.
    If you have number of microphones != from number of speakers this class still works.
    """

    def __init__(
            self,
            n_M: int = 2,
            n_L: int = 2,
            fs: int = 32000,
            nfft: int = 2 * 480000,
            alias_decay_db: float = 0.0,
            FIR_order: int = 100,
            requires_grad: bool = False,
            wgn_t60: float = 2.0
    ) -> None:
        r"""
        Initializes the class as a series of FIRs and WGN reverb.

            **Args**:
                - n_M (int): Number of system microphones.
                - n_L (int): Number of system loudspeakers.
                - fs (int): Sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): Anti-time-aliasing decay [dB].
                - FIR_order (int): FIR filter order.
                - wgn_t60 (float): T60 of the WGN reverb (broadband).
                - requires_grad (bool): Whether the filter is learnable.

            **Attributes**:
                [- _VrRoom attributes]
                - FIR_order (int): FIR filter order.
                - wgn_t60 (float): T60 of the WGN reverb (broadband).
        """

        VrRoom.__init__(
            self,
            n_M=n_M,
            n_L=n_L,
            fs=fs,
            nfft=nfft,
            alias_decay_db=alias_decay_db
        )
        self.FIR_order = FIR_order
        self.wgn_t60 = wgn_t60

        self.FIRs = self.gen_FIRs(requires_grad)
        self.WGN_rev = self.gen_WGN_rev()

        self.v_ML = system.Series(self.FIRs, self.WGN_rev)

    def gen_FIRs(self, requires_grad) -> dsp.Filter:
        module = dsp.Filter(
            size=(self.FIR_order, self.n_L, self.n_M),
            nfft=self.nfft,
            requires_grad=requires_grad,
            alias_decay_db=self.alias_decay_db
        )
        return module

    def gen_WGN_rev(self) -> dsp.parallelFilter:
        rirs = flamo.functional.WGN_reverb(
            matrix_size=(self.n_L,),
            t60=self.wgn_t60,
            samplerate=self.fs,
        )
        module = dsp.parallelFilter(
            size=(rirs.shape[0], self.n_L),
            nfft=self.nfft,
            requires_grad=False,
            alias_decay_db=self.alias_decay_db
        )
        return module

    def add_reverb_to_chain(self) -> None:
        if self.in_training: Warning("Training has to be completed by now")
        self.in_training = False
        self.v_ML = system.Series(
            self.FIRs,
            self.WGN_rev
        )
        return None

def train_virtual_room(args) -> None:
    # -------------------- Initialize RES ---------------------
    # Time-frequency
    samplerate = 32000  # Sampling frequency in Hz
    nfft = 2 * 480000#int(samplerate * 4.5)  # FFT size
    alias_decay_db = 0  # Anti-time-aliasing decay in dB

    # Physical room
    dataset_directory = "./data"
    room_name = "SmallSystem"

    physical_room = PhRoom_dataset(
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        dataset_directory=dataset_directory,
        room_name=room_name
    )
    n_M = physical_room.transducer_number['mcs']  # Number of microphones
    n_L = physical_room.transducer_number['lds']  # Number of loudspeakers

    # Virtual room
    wgn_reverb_t60_broadband = 0.5  # T60 of the WGN reverb
    virtual_room = VrRoom_FIRS_WGN(
        n_M=n_M,
        n_L=n_L,
        fs=samplerate,
        nfft=nfft,
        alias_decay_db=alias_decay_db,
        FIR_order=args.fir_length,
        requires_grad=True,
        wgn_t60=wgn_reverb_t60_broadband
    )

    # Reverberation Enhancement System
    res = RES(
        physical_room=physical_room,
        virtual_room=virtual_room,
        loop_gain_dB=args.loop_gain_dB
    )

    if args.load_from_state:
        res.set_v_ML_state(torch.load(f"model_states/{args.load_state_filename}"))

    # ------------------- Model Definition --------------------
    model = system.Shell(
        core=res.open_loop(),
        input_layer=system.Series(
            dsp.FFT(nfft=nfft),
            dsp.Transform(lambda x: x.diag_embed())
        )
    )

    # ------------- Performance at initialization -------------
    evs_init = res.open_loop_eigenvalues()
    _, _, ir_init = res.system_simulation()

    # ----------------- Initialize dataset --------------------
    dataset_input = torch.zeros(1, samplerate, n_M)
    dataset_input[:, 0, :] = 1
    dataset_target = system_equalization_curve(evs=evs_init, fs=samplerate, nfft=nfft, f_c=8000)
    dataset_target = dataset_target.view(1, -1, 1).expand(1, -1, n_M)

    dataset = Dataset(
        input=dataset_input,
        target=dataset_target,
        expand=args.num,
        device=args.device
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=1, split=args.split, shuffle=False)

    # ------------------- Initialize Trainer ------------------
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device
    )

    # ---------------- Initialize Loss Function ---------------
    criterion = MSE_evs_mod(
        iter_num=args.num,
        freq_points=nfft // 2 + 1,
        samplerate=samplerate,
        lowest_f=20.0,
        highest_f=samplerate / 2
    )
    trainer.register_criterion(criterion, 1.0)

    # --------------- Train the model and save ---------------
    if not args.load_from_state:
        trainer.train(train_loader, valid_loader)
        res.save_state_to(directory='model_states/')

    # ------------ Performance after optimization ------------
    evs_opt = res.open_loop_eigenvalues()
    _, _, ir_opt = res.system_simulation()
    res.set_G(db2mag(mag2db(res.compute_GBI()) - 6.0))
    _, _, ir_opt_added_gain = res.system_simulation()

    # ------------------------ Plots -------------------------
    plot_evs_compare(evs_init, evs_opt, samplerate, nfft, 20, 8000)
    plot_irs_compare(ir_init[:, 0], ir_opt[:, 0], samplerate)
    plot_spectrograms_compare(ir_init[:, 0], ir_opt[:, 0], ir_opt_added_gain[:, 0], samplerate, nfft=2**11, noverlap=2**10)

    return None


###########################################################################################

if __name__ == '__main__':

    # Define training pipeline hyperparameters
    parser = argparse.ArgumentParser()

    # ----------------------- Dataset ----------------------
    parser.add_argument('--num', type=int, default=200, help='dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation')
    parser.add_argument('--split', type=float, default=0.9, help='split ratio for training and validation')
    # ---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=1e-5,
                        help='Minimum improvement in validation loss to be considered as an improvement')
    # ---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # ------------------------ AAES ------------------------
    parser.add_argument('--loop_gain_dB', type=float, default=-3.0, help='loop gain in decibels')
    parser.add_argument('--fir_length', type=int, default=100, help='number of FIR taps per channel')
    # ----------------- Train or Load State ----------------
    parser.add_argument('--load_from_state', type=bool, default=False, help='should load from state and skip training')
    parser.add_argument('--load_state_filename', type=str, default="2026-03-24_14.55.56.pt", help='model state filename to load')
    # ----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # make training output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('training_output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Run script
    train_virtual_room(args)
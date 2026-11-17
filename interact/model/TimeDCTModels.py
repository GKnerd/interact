''' InteRACT with a per-step output head: DCT coefficients over time (ablation run 1) '''
import torch
import torch.nn as nn
import torch_dct as dct

from interact.model.Models import IntentInformedForecaster


class TimeDCTForecaster(IntentInformedForecaster):
    '''
    IntentInformedForecaster with a different output head; encoders and decoder are unchanged.

    The decoder vector is mapped to horizon_len x (joints * 3) DCT coefficients over time, an
    inverse DCT over time gives per-frame offsets, and the offsets are added to the last history
    frame. Every step and coordinate gets its own value from the input, instead of one direction
    scaled by a fixed speed profile.
    '''

    def __init__(self, d_model=64, alice_joints_num=9, horizon_len=15, **kwargs):
        super().__init__(d_model=d_model, alice_joints_num=alice_joints_num, **kwargs)
        self.horizon_len = horizon_len
        self.n_coords = alice_joints_num * 3
        # 15 x 27 numbers from the decoder vector: one per (frequency, coordinate)
        self.time_dct_head = nn.Linear(d_model, horizon_len * self.n_coords)
        # the parent's init loop ran before this layer existed
        nn.init.xavier_uniform_(self.time_dct_head.weight)

    def _head(self, dec_output, alice_hist):
        coeffs = self.time_dct_head(self.decoder_linear(dec_output))  # (B, 1, 15*27)
        coeffs = coeffs.view(-1, self.horizon_len, self.n_coords)     # (B, 15, 27) row k = frequency k
        # torch_dct works on the last axis: transpose so the inverse DCT runs over time
        offsets = dct.idct(coeffs.transpose(1, 2), norm='ortho')      # (B, 27, 15)
        offsets = offsets.transpose(1, 2)                             # (B, 15, 27) row t = frame t
        return alice_hist[:, -1:] + offsets                           # offsets from the last seen frame

    def forward(self,
            alice_hist,
            bob_hist,
            bob_future,
            add_spe=True,
            bob_is_robot=False):
        # Same as IntentInformedForecaster.forward up to the decoder; only the head differs.

        ### Index out bob's relevant joints
        if self.bob_joint_indices is not None:
            bob_future = bob_future[:,:,self.bob_joint_indices]
            bob_hist = bob_hist[:,:,self.bob_joint_indices]

        ### Intent informed forecasting only cares about Bob's position at final timestep
        bob_future = bob_future[:,-1:]

        ### local history encoding
        alice_displacement = alice_hist[:,1:alice_hist.shape[1],:]-alice_hist[:,:alice_hist.shape[1]-1,:]
        alice_displacement_dct = dct.dct(alice_displacement)
        alice_local_enc = self.alice_local_hist_encoder(alice_displacement_dct)
        alice_local_output, *_ = self.encoder(alice_local_enc, 1, None)

        ### global history encoding
        alice_global_enc = self.alice_global_hist_encoder(alice_hist)
        if not self.one_hist:
            bob_global_enc = self.bob_global_hist_encoder(bob_hist)
            global_enc = torch.cat([alice_global_enc, bob_global_enc],dim=1)
        else:
            global_enc = alice_global_enc
        global_output, *_ = self.encoder_global(global_enc,
                    1 if self.one_hist else 2,
                    src_mask = None,
                    global_feature=True)

        ### conditional future encoder
        if not self.conditional_forecaster:
            bob_future.fill_(0)
        bob_cond_future_enc = self.bob_global_future_encoder(bob_future)

        spe = 0
        if add_spe:
            alice_spe = torch.norm(alice_hist-alice_hist[:, -1].unsqueeze(1), dim=-1)
            bob_spe = torch.norm(bob_hist-alice_hist[:, -1, self.bob_joint_indices].unsqueeze(1), dim=-1)
            spe = torch.exp(-torch.cat([alice_spe, bob_spe] if not self.one_hist else [alice_spe], dim=1)).unsqueeze(2)

        encoder_output = torch.cat([alice_local_output, global_output+spe], dim=1)

        dec_output, dec_attention, *_ = self.decoder(bob_cond_future_enc, None, encoder_output, None)
        return self._head(dec_output, alice_hist)

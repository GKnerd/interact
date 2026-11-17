''' TimeDCTForecaster with a DCT over time on the local history input (ablation run 3) '''
import torch
import torch_dct as dct

from interact.model.TimeDCTModels import TimeDCTForecaster


class TimeDCTInputForecaster(TimeDCTForecaster):
    '''
    TimeDCTForecaster whose local branch takes the history displacements as frequencies over time.

    The original dct.dct works on the last axis and so mixes the 27 coordinates of each frame;
    right before a linear layer that has no effect. Here the DCT runs over the 14 displacement
    frames: token k of the local encoder is frequency k of the motion (k = 0: mean velocity over
    the history), for all coordinates. Global branch, head and modules are those of
    TimeDCTForecaster.
    '''

    def forward(self,
            alice_hist,
            bob_hist,
            bob_future,
            add_spe=True,
            bob_is_robot=False):
        # Same as TimeDCTForecaster.forward except for the DCT of the local history encoding.

        ### Index out bob's relevant joints
        if self.bob_joint_indices is not None:
            bob_future = bob_future[:,:,self.bob_joint_indices]
            bob_hist = bob_hist[:,:,self.bob_joint_indices]

        ### Intent informed forecasting only cares about Bob's position at final timestep
        bob_future = bob_future[:,-1:]

        ### local history encoding
        alice_displacement = alice_hist[:,1:alice_hist.shape[1],:]-alice_hist[:,:alice_hist.shape[1]-1,:]
        # DCT over time: (B, 14, 27) -> (B, 27, 14) -> DCT -> (B, 14, 27), row k = frequency k
        alice_displacement_dct = dct.dct(alice_displacement.transpose(1, 2), norm=None).transpose(1, 2)
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

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib.pyplot as plt
import os
from matplotlib.cbook import sanitize_sequence
from matplotlib.animation import FuncAnimation, writers, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from tqdm import tqdm

class AnimationRenderer:

    def __init__(self, skeleton, poses_generator, algos, t_hist, t_pred, baselines=['gt', 'context'], 
                        fix_0=True, output_dir="./out", size=6, ncol=5, bitrate=-1, gif_dpi=150,
                        type="3d"):
        
        os.makedirs(output_dir, exist_ok=True)

        self.skeleton = skeleton
        self.poses_generator = poses_generator
        self.algos = algos
        self.t_hist = t_hist
        self.t_pred = t_pred
        self.baselines = baselines
        self.fix_0 = fix_0
        self.output_dir = output_dir
        self.ncol = ncol
        self.size = size
        self.bitrate = bitrate
        self.gif_dpi = gif_dpi
        self.t0_digit_pressed = time.time()
        self.digits_pressed = []
        assert type.lower() in ["3d", "2d"], f"'{type}' is not a supported visualization type"
        self.fig_3d = type.lower() == "3d"
        self.show_hist = True

        self.fps = self.skeleton.fps
        self.elev, self.azim, self.roll = self.skeleton.get_camera_params()
        self.all_poses, self.sample_idx = next(poses_generator)
        
        # all_poses is a dictionary with results for each model/gt/baseline
        self.algo = algos[-1] if len(algos) > 0 else next(iter(self.all_poses.keys()))
        self.t_total = next(iter(self.all_poses.values())).shape[0]

        # filter poses so that match 'gt', 'context' or the name of the first algorithm selected
        self.poses_dict = dict(filter(lambda x: x[0] in baselines or self.algo == x[0].split('_')[0], self.all_poses.items()))
        self.poses = list(self.poses_dict.values()) 

        self.anim = None
        self.initialized = False
        self.animating = False
        self.gif_writer = PillowWriter(fps=self.fps, bitrate=-1, codec="libx264")

    def run_animation(self, show=True):
        plt.ioff()
        # Create a single figure
        self.fig = plt.figure(figsize=(10, 10))
        
        # Setup 3D Skeleton Plot
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self.ax_3d.view_init(elev=self.elev, azim=self.azim, roll=self.roll)
        self.radius, x_r, y_r, z_r = self.skeleton.get_3d_radius()
        self.ax_3d.set_xlim3d(x_r)
        self.ax_3d.set_ylim3d(y_r)
        self.ax_3d.set_zlim3d(z_r)
        self.ax_3d.set_aspect('auto')
        self.ax_3d.set_axis_off()
        
        # Text Box for MPJPE and Frame Info 
        self.error_text = self.ax_3d.text2D(0.05, 0.95, "", transform=self.ax_3d.transAxes, 
                                            fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        self.axes = [self.ax_3d]
        
        # Track GT lines, Prediction lines, and Deviation lines
        self.lines = {"gt": [], "pred": [], "deviation": []}
        
        # Precompute MPJPE
        self.gt_data = self.poses_dict['gt']
        self.pred_data = self.poses_dict[self.algo]
        diff = self.pred_data - self.gt_data
        self.errors = np.linalg.norm(diff, axis=-1).mean(axis=(1, 2)) * 1000 # Convert to mm
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.animating = True
        self._show_animation()
        if show:
            plt.show()
        return self.anim

    def store_all(self, type="gif", idx=-1):
        assert type.lower() in ["gif", "mp4"], f"'{type}' is not a supported storage output type"
        self.algo = self.algos[idx]

        writer = self.gif_writer if type == "gif" else self.mp4_writer
        ext = ".gif" if type == "gif" else ".mp4"
        kwargs = {"dpi": self.gif_dpi} if type == "gif" else {}

        stored_paths = []
        os.makedirs(self.output_dir, exist_ok=True)

        self.run_animation(show=False)
        self._reload_poses()
        output_path = os.path.join(self.output_dir, f"%s_%s{ext}" % (self.algo, self.sample_idx))
        
        try:
            self.anim.save(output_path, writer=writer, progress_callback=None, **kwargs)
        except:
            # ffmpeg does not have the appropriate encoding algorithm installed => try libopenh264
            self.mp4_writer = writers['ffmpeg'](fps=self.fps, metadata={}, bitrate=self.bitrate, codec='libopenh264', extra_args=['-pix_fmt', 'yuv420p'])
            writer = self.mp4_writer
            self.anim.save(output_path, writer=writer, progress_callback=None, **kwargs)
        stored_paths.append(output_path)

        for all_poses, sample_idx in tqdm(self.poses_generator):
            self.all_poses, self.sample_idx = all_poses, sample_idx
            self._reload_poses()
            self._show_animation()

            output_path = os.path.join(self.output_dir, f"%s_%s{ext}" % (self.algo, self.sample_idx))
            self.anim.save(output_path, writer=writer, progress_callback=None, **kwargs)
            stored_paths.append(output_path)

        return stored_paths

    def _update_video(self, i):
        if self.show_hist:
            i = i % (self.t_hist + self.t_pred)
        else:
            i = self.t_hist + i % (self.t_pred)
            
        self.fig.suptitle(f'Sample {self.sample_idx}', fontsize=14, fontweight='bold')

        parents = self.skeleton.parents()

        if not self.initialized:
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                col = self.skeleton.get_color(j)
                
                # GT lines (Black and transparent, as the anchor)
                gt_line = self.ax_3d.plot([0, 0], [0, 0], [0, 0], zdir='z', c='black', alpha=0.3, linestyle='--')[0]
                self.lines["gt"].append(gt_line)
                
                # Prediction lines (Colored and solid)
                pred_line = self.ax_3d.plot([0, 0], [0, 0], [0, 0], zdir='z', c=col, linewidth=2.5)[0]
                self.lines["pred"].append(pred_line)
            
            # Deviation lines (Connecting GT joints to Pred joints to show error direction)
            for j in range(self.gt_data.shape[2]): 
                dev_line = self.ax_3d.plot([0, 0], [0, 0], [0, 0], zdir='z', c='gray', linestyle=':', linewidth=1.0)[0]
                self.lines["deviation"].append(dev_line)

            self.initialized = True

        else:
            counter = 0
            pos_gt = self.gt_data[i, 0]
            pos_pred = self.pred_data[i, 0]

            # ==========================================
            # NEW: DYNAMIC CAMERA TRACKING & ZOOM
            # ==========================================
            # Center the camera on the root joint (joint 0) to track movement
            root_pos = pos_gt[0]
            zoom_radius = 0.6  # Change this to zoom in/out! (Smaller number = bigger human)
            
            self.ax_3d.set_xlim3d([root_pos[0] - zoom_radius, root_pos[0] + zoom_radius])
            self.ax_3d.set_ylim3d([root_pos[1] - zoom_radius, root_pos[1] + zoom_radius])
            self.ax_3d.set_zlim3d([root_pos[2] - zoom_radius, root_pos[2] + zoom_radius])
            # ==========================================

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                # Always update GT
                self.lines["gt"][counter].set_data_3d(
                    [pos_gt[j, 0], pos_gt[j_parent, 0]], 
                    [pos_gt[j, 1], pos_gt[j_parent, 1]], 
                    [pos_gt[j, 2], pos_gt[j_parent, 2]]
                )
                
                # Only show prediction during prediction phase
                if i >= self.t_hist:
                    self.lines["pred"][counter].set_data_3d(
                        [pos_pred[j, 0], pos_pred[j_parent, 0]], 
                        [pos_pred[j, 1], pos_pred[j_parent, 1]], 
                        [pos_pred[j, 2], pos_pred[j_parent, 2]]
                    )
                else:
                    self.lines["pred"][counter].set_data_3d([], [], [])
                    
                counter += 1

            # Update Deviation lines only during prediction
            if i >= self.t_hist:
                for j in range(self.gt_data.shape[2]):
                    self.lines["deviation"][j].set_data_3d(
                        [pos_gt[j, 0], pos_pred[j, 0]],
                        [pos_gt[j, 1], pos_pred[j, 1]],
                        [pos_gt[j, 2], pos_pred[j, 2]]
                    )
            else:
                for j in range(self.gt_data.shape[2]):
                    self.lines["deviation"][j].set_data_3d([], [], [])

        # Update the Text Box Overlay
        if i >= self.t_hist:
            self.error_text.set_text(f"Phase: PREDICTION\nFrame: +{i - self.t_hist + 1}\nMPJPE: {self.errors[i]:.1f} mm")
            self.error_text.set_color("red")
        else:
            self.error_text.set_text(f"Phase: HISTORY\nFrame: {i + 1}/{self.t_hist}\nMPJPE: 0.0 mm")
            self.error_text.set_color("black")

    def _reload_poses(self):
        self.poses_dict = dict(filter(lambda x: x[0] in self.baselines or self.algo == x[0].split('_')[0], self.all_poses.items()))
        
        self.gt_data = self.poses_dict['gt']
        self.pred_data = self.poses_dict[self.algo]
        
        diff = self.pred_data - self.gt_data
        self.errors = np.linalg.norm(diff, axis=-1).mean(axis=(1, 2)) * 1000

        if self.fig_3d:
            # Re-center the camera on the new sequence's root
            trajectory = self.gt_data[0, 0, 0, [0, 1, 2]] 
            self.ax_3d.set_xlim3d([-self.radius/2 + trajectory[0], self.radius/2 + trajectory[0]])
            self.ax_3d.set_ylim3d([-self.radius/2 + trajectory[1], self.radius/2 + trajectory[1]])
            self.ax_3d.set_zlim3d([-self.radius/2 + trajectory[2], self.radius/2 + trajectory[2]])

    def _show_animation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, self._update_video, frames=np.arange(0, (self.t_hist + self.t_pred)), interval=1000 / self.fps, repeat=True)
        plt.draw()
        self.animating = True

    def _save_figs(self):
        old_algo = self.algo
        for algo in self.algos:
            self.algo = algo
            self._reload_poses()
            self._update_video(self.t_total - 1)
            self.fig.savefig(os.path.join(self.output_dir, '%s_%d.png' % (algo, self.sample_idx)), dpi=self.gif_dpi, transparent=True)
        self.algo = old_algo

    def _pause_animation(self, finish=False):
        if self.animating:
            self.animating = False
            self.anim.event_source.stop()
            if finish:
                self.anim = None

    def _resume_animation(self):
        if self.animating:
            self.animating = True
            self.anim.event_source.start()

    def _switch_animating(self):
        self.animating = not self.animating
        if self.animating:
            self.anim.event_source.stop()
        else:
            self.anim.event_source.start()

    def _on_key(self, event):
        if event.key == 'n':
            self.all_poses, self.sample_idx = next(self.poses_generator)
            self._reload_poses()
            self._show_animation()
        elif event.key == 'h': # do not show history
            self.show_hist = not self.show_hist
            self._reload_poses()
            self._show_animation()
        elif event.key == 'p':
            if self.animating:
                self._pause_animation()
                self.animating = False
            else:
                self._show_animation()
                self.animating = True
        elif event.key == ' ':
            self._switch_animating()
        elif event.key in ['g', 'v']:
            self._pause_animation(finish=True)
            mode = "gif" if event.key == 'g' else "mp4"
            self._save(mode)
            self._show_animation()
        elif event.key == 'i':  # save images
            self._pause_animation(finish=True)
            self._save_figs()
            self._reload_poses()
            self._show_animation()
        elif event.key == 'q' or event.key == "escape":  
            exit()
        elif event.key.isdigit() and len(self.algos) > 1 and int(event.key) < len(self.algos): 
            current_time = time.time()
            idx = int(np.sum([self.digits_pressed[i] * 10 ** (len(self.digits_pressed) - i) for i in range(len(self.digits_pressed))])) + int(event.key)
            continue_seq = (current_time - self.t0_digit_pressed < 1) 
            if idx >= len(self.algos) or not continue_seq: 
                idx = int(event.key)
                self.digits_pressed = [idx]
            else:
                self.digits_pressed.append(int(event.key))

            self.algo = self.algos[idx]
            self._reload_poses()
            self._show_animation()
            self.t0_digit_pressed = current_time

    def _save(self, mode):
        self.anim = FuncAnimation(self.fig, self._update_video, frames=np.arange(0, (self.t_hist + self.t_pred)), interval=1000 / self.fps, repeat=False)
        if mode == 'mp4':
            output_path = os.path.join(self.output_dir, "%s_%d.mp4" % (self.algo, self.sample_idx))
            print(f"Saving to '{output_path}'...")
            self.anim.save(output_path, writer=self.mp4_writer)
        elif mode == 'gif':
            output_path = os.path.join(self.output_dir, "%s_%d.gif" % (self.algo, self.sample_idx))
            print(f"Saving to '{output_path}'...")
            self.anim.save(output_path, dpi=self.gif_dpi, writer=self.gif_writer)
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {self.output_dir}!')

    def _on_move(self, event):
        # We only have one 3D axis now, so we don't need to manually sync rotations 
        # across multiple subplots. Matplotlib handles the native 3D rotation automatically!
        pass

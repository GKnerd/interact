
import numpy as np


# https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=3
CBS1 = "#000000"
CBS2_hist = "#4dac26"  # green
CBS3_hist = "#d01c8b"  # pink


CBS1 = "#000000" # black
CBS2 = "#3d65a5" # blue
CBS3 = "#e57a77" # orange

class Skeleton:
    def __init__(self, parents, joints, colors, dim='3d', fps=25, colors_hist=None):
        #assert len(joints_left) == len(joints_right)
        
        self._parents = np.array(parents)
        self._joints = joints
        self._colors = colors
        self._colors_hist = colors_hist if colors_hist is not None else colors
        self.dim = dim
        self.fps = fps
        self._compute_metadata()
    
    def num_joints(self):
        return len(self._parents)
    
    def parents(self):
        return self._parents
    
    def has_children(self):
        return self._has_children
    
    def children(self):
        return self._children

    def get_color(self, idx):
        return self._colors[idx]

    def get_color_hist(self, idx):
        return self._colors_hist[idx]

    def get_3d_radius(self):
        raise NotImplementedError()

    def get_camera_params(self):
        # returns elev, azim, roll degrees
        return 15, 0, 0 # best for h36m
    
    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        self._colors = [col for i, col in enumerate(self._colors) if i not in joints_to_remove]
        self._colors_hist = [col for i, col in enumerate(self._colors_hist) if i not in joints_to_remove]
        
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]
                
        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)
        
        
        if self._joints is not None:
            new_joints = []
            for joint in self._joints:
                if joint in valid_joints:
                    new_joints.append(joint - index_offsets[joint])
            self._joints = new_joints

        self._compute_metadata()
        
        return valid_joints
    
    def joints_left(self):
        return self._joints
        
    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)



class SkeletonH36M(Skeleton):
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self.joints_left = joints_left
        self.joints_right = joints_right
        
        colors = [CBS1 for i in range(len(parents))]
        colors_hist = [CBS1 for i in range(len(parents))]
        for i in self.joints_left:
            colors[i] = CBS2
            colors_hist[i] = CBS2_hist
        for i in self.joints_right:
            colors[i] = CBS3
            colors_hist[i] = CBS3_hist

        super().__init__(parents=parents,
                                 joints=list(sorted(joints_left + joints_right)),
                                 colors=colors,
                                 dim='3d',
                                 fps=50,
                                 colors_hist=colors_hist
                                 )
        self.xlim = (-0.5, 0.5)
        self.ylim = (-0.5, 1.5)

    def get_3d_radius(self):
        r = 5
        x = [-r/2, r/2]
        y = [-r/2, r/2]
        z = [-r/2, r/2]
        return r, x, y, z


class SkeletonAMASS(Skeleton):
    """
    - Hips # 0
    - LeftUpLeg # 1
    - RightUpLeg # 2
    - Spine0 # 3
    - LeftLeg # 4
    - RightLeg # 5
    - Spine1 # 6
    - LeftFoot # 7
    - RightFoot # 8
    - Spine2 # 9
    - LeftToeBase # 10
    - RightToeBase # 11
    - Neck # 12
    - LeftShoulder # 13
    - RightShoulder # 14
    - Head # 15
    - LeftArm # 16
    - RightArm # 17
    - LeftElbow # 18
    - RightElbow # 19
    - LeftHand # 20
    - RightHand # 21
    """
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self.joints_left = joints_left
        self.joints_right = joints_right
        
        colors = [CBS1 for i in range(len(parents))]
        colors_hist = [CBS1 for i in range(len(parents))]
        for i in self.joints_left:
            colors[i] = CBS2
            colors_hist[i] = CBS2_hist
        for i in self.joints_right:
            colors[i] = CBS3
            colors_hist[i] = CBS3_hist

        super().__init__(parents=parents,
                                 joints=list(sorted(joints_left + joints_right)),
                                 colors=colors,
                                 dim='3d',
                                 fps=50,
                                 colors_hist=colors_hist
                                 )
        self.xlim = (-0.5, 0.5)
        self.ylim = (-0.5, 1.5)

    def get_3d_radius(self):
        r = 5
        x = [-r/2, r/2]
        y = [-r/2, r/2]
        z = [-r/2, r/2]
        return r, x, y, z


# class SkeletonCoMaDHR(Skeleton):
#     def __init__(self):
#         # Alice uses 9 joints. 
#         # Logical hierarchy for upper body: Pelvis(0), Spine(1), Neck/Head(2), L_Shoulder(3), L_Elbow(4), L_Hand(5), R_Shoulder(6), R_Elbow(7), R_Hand(8)
#         alice_parents = [-1, 0, 1, 1, 3, 4, 1, 6, 7] 
        
#         # Robot uses 2 joints. 
#         # It sits at index 9 and 10 in the unified array.
#         robot_parents = [-1, 9] 
        
#         unified_parents = alice_parents + robot_parents
#         total_joints = len(unified_parents)

#         # Colors setup
#         colors = ["#000000" for _ in range(total_joints)]
#         colors_hist = ["#000000" for _ in range(total_joints)]
        
#         # Alice Left (Blue) / Right (Orange)
#         joints_left = [3, 4, 5]
#         joints_right = [6, 7, 8]
#         for i in joints_left:
#             colors[i] = "#3d65a5"
#             colors_hist[i] = "#4dac26" # Green for history
#         for i in joints_right:
#             colors[i] = "#e57a77"
#             colors_hist[i] = "#d01c8b" # Pink for history
            
#         # Robot colors (Gold)
#         colors[9] = "#FFD700" 
#         colors[10] = "#FFD700"
#         colors_hist[9] = "#B8860B"
#         colors_hist[10] = "#B8860B"

#         super().__init__(parents=unified_parents,
#                          joints=list(range(total_joints)),
#                          colors=colors,
#                          dim='3d',
#                          fps=15, 
#                          colors_hist=colors_hist)
        
#         self.xlim = (-1.5, 1.5)
#         self.ylim = (-1.5, 1.5)

#     def get_3d_radius(self):
#         r = 3.0
#         return r, [-r/2, r/2], [-r/2, r/2], [-r/2, r/2]
    
#     def get_camera_params(self):
#         return 20, 45, 0


class SkeletonCoMaDHR(Skeleton):
    def __init__(self):
        # 9 Human joints: Upper Back(0), L_Shoulder(1), L_Elbow(2), L_Wrist(3), L_Hand(4), R_Shoulder(5), R_Elbow(6), R_Wrist(7), R_Hand(8)
        human_parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7] 
        
        # 2 Robot joints: Base/Wrist(9), End-effector/Hand(10)
        robot_parents = [-1, 9] 
        
        unified_parents = human_parents + robot_parents
        total_joints = len(unified_parents)

        colors = ["#000000" for _ in range(total_joints)]
        colors_hist = ["#000000" for _ in range(total_joints)]
        
        # Left Arm (Blue)
        joints_left = [1, 2, 3, 4]
        for i in joints_left:
            colors[i] = "#3d65a5"
            colors_hist[i] = "#4dac26" 
            
        # Right Arm (Orange)
        joints_right = [5, 6, 7, 8]
        for i in joints_right:
            colors[i] = "#e57a77"
            colors_hist[i] = "#d01c8b" 
            
        # Robot (Gold)
        colors[9] = "#FFD700" 
        colors[10] = "#FFD700"
        colors_hist[9] = "#B8860B"
        colors_hist[10] = "#B8860B"

        super().__init__(parents=unified_parents,
                         joints=list(range(total_joints)),
                         colors=colors,
                         dim='3d',
                         fps=15, 
                         colors_hist=colors_hist)
        
        self.xlim = (-1.5, 1.5)
        self.ylim = (-1.5, 1.5)

    def get_3d_radius(self):
        r = 3.0
        return r, [-r/2, r/2], [-r/2, r/2], [-r/2, r/2]
    
    def get_camera_params(self):
        return 20, 45, 0

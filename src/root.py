# Root object 

import numpy as np
import xmltodict
from math import sqrt, atan2, pi
import scipy as sp

np.random.seed(19680801)

interval = 6

# Now given that the time interval is variable, we use coord_th instead;
# When coord_th has issues, we use coord_t to correct for it.
# We need to reasign it.

class Node:
    """Class reprsenting a point on a root structure.
    
    Attributes:
        x: A float indicating X position in mm.
        y: A float indicating Y position in mm.
        time: A float indicating time of appiration in hr.
        length: A float indicating length from first node in branch to this node in mm
            or `None` if not yet set.
        speed: A float indicating average speed of travel from previous node to
            this in mm/hr or `None` if not yet set.
        angle: A float indicating the angle from previous node to this in radians 
            or `None` if not yet set
        is_tip: A boolean indicating if this node was a branch tip in any frame
            or `None` if not yet set.
        index: A integer indicating the position of this node in it's branch node list
            or `None` if not yet set.
    """

    def __init__(self, x: float, y: float, time: float, hr: float) -> None:
        """Initializes the instance based on spatial position and frame of first appearance.

        Args:
            x: float indicating X position in mm
            y: float indicating Y position in mm
            time: float indicating time of appiration
        """
        self.x = x
        self.y = y
        self.hr = hr
        self.time = time
        self.length = None
        self.speed = None
        self.angle = None
        self.is_tip = None
        self.index = None
    
    def __str__(self) -> str:
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.time) + ")"
    
    def angle_to(self, node2):
        dx = node2.x - self.x
        dy = node2.y - self.y
        return atan2(-dx,dy)/pi*180
        

class Branch:
    """ Class representing branch in a root system.
         
    Attributes:
        nodes: A list of Nodes indicating the set of recorded positions on the branch
            ordered from root.
        tips: A list containing subset of `nodes` which are the branch tip in any frame.
        appiration: A float indicating the earliest time in hrs of any node in nodes.
        depth: A float indicating the depth in mm of this branching point
            or None if this is the primary root or this is the 
    """

    def __init__(self, nodes: list, pxl_len_mm=1.0):
        """Initializes the instance based on a length scale.

        Args:
            nodes: A list of Nodes or a list of RSML style dictionaries
                indicating observed points on the branch ordered from root.
            pxl_len_mm: A float indicating the mm:pxl ratio to convert 
                measurements in `nodes` if `nodes` is a dict.
        """
        assert type(nodes) == list, nodes

        self.nodes = []
        self.tips = None
        self.appiration = None 
        self.depth = None
        self.angle = None

        if nodes:
            if type(nodes[0]) == Node:
                self.nodes = nodes
            else:
                self.nodes = []
                for n in nodes:
                    coord_t = float(n['@coord_t'])
                    if coord_t < 2:
                        time = 0.0
                    else:
                        # time is still needed in case coord_th is messed up
                        time = (coord_t - 1) * interval
                    node = Node(x = float(n['@coord_x']) * pxl_len_mm,
                                y = float(n['@coord_y']) * pxl_len_mm, 
                                time = time,
                                hr = float(n['@coord_th'])
                                ) 
                    self.nodes.append(node)
            self.set_data()

    def set_data(self):
        self.set_node_indices()
        self.set_tips()
        self.set_appiration()
        self.set_lengths()
        self.set_speeds()
        self.set_angles()

    def set_node_indices(self) -> None:
        for i in range(len(self.nodes)):
            self.nodes[i].index = i

    def is_tip(self, tip: Node) -> bool:
        '''Returns bool of whether ith element in self.nodes is a tip'''
        i = tip.index

        # the last node:
        if i == len(self.nodes) - 1:
            return True
        # if it is the day 0 existing nodes, then the last one with the coord_th = 0 
        elif self.nodes[i].hr == 0 and self.nodes[i].hr != self.nodes[i+1].hr:
            return True
        # if it is not day 0 existing nodes, then it's the later full coord_t nodes
        elif self.nodes[i].hr != 0 and self.nodes[i].time == round(self.nodes[i].time):
            return True

    def set_tips(self) -> None:
        '''Set branch.tips attribute. Set is_tip attributes of all nodes.'''
        for node in self.nodes:
            node.is_tip = self.is_tip(node)
        self.tips = [node for node in self.nodes if node.is_tip]

    def set_appiration(self) -> None:
        '''Set appiration attribute of branch.'''
        # self.appiration = min(self.nodes, key=lambda x: x.time).time
        self.appiration = min(self.nodes, key=lambda x: x.hr).hr

    def set_lengths(self) -> None:
        '''Set length attribute of each node.'''
        self.nodes[0].length = 0
        for i in range(len(self.nodes)-1):
            prev = self.nodes[i]
            curr = self.nodes[i+1]
            dist_traveled = sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)
            curr.length = prev.length + dist_traveled
    
    def set_speeds(self) -> None:
        '''Set speed attribute of each tip except first.'''
        self.tips[0].speed = None
        for i in range(1, len(self.tips)):
            prev = self.tips[i-1]
            curr = self.tips[i]
            if prev.hr == curr.hr:
                curr.speed = None
            else:
                curr.speed = (curr.length - prev.length) / (curr.hr - prev.hr)

            
    def set_angles(self) -> None:
        '''Set angle attribute of each node except first.'''
        self.nodes[0].angle = None
        for i in range(1, len(self.nodes)):
            curr = self.nodes[i]
            curr.angle = self.nodes[i-1].angle_to(curr)

class Root:
    """Class representing a root system.
    
    Attributes:
        primary: A Branch that is the primary root.
        laterals: A list of Branch objects that are all lateral roots.
        branch_pts: A list of tuples (i, j) of integer i and j in [0,1]. 
            ex) If the branching point of the 4th lateral is 0.5 of the 
            distance between the jth and j+1th node of the primary,
            then the ith tuple in self.branch_pts = (i, 0.5)
    """

    def __init__(self, primary, laterals):

        self.primary = primary
        self.laterals = laterals if laterals else None
        self.length = None
        self.branch_pts = None
        self.set_data()

    def set_data(self):
        if self.laterals is not None:
            self.set_branch_pts()
            self.set_depths()
            self.set_branch_angles()

    def set_branch_pts(self) -> None: 
        
        '''Set attribute self.branch_pts.'''
        n_ints = 100 # Number of interpolant pts between each primary node
        primary_pts = []
        for i in range(len(self.primary.nodes) - 1):
            x1 = self.primary.nodes[i].x
            y1 = self.primary.nodes[i].y
            x2 = self.primary.nodes[i+1].x
            y2 = self.primary.nodes[i+1].y
            for j in range(n_ints):
                primary_pts.append([x1 + j * (x2 - x1) / n_ints,
                                   y1 + j * (y2 - y1) / n_ints]) # Primary nodes with interpolation
                
        # Lateral root first nodes
        lr_pts = [(lr.nodes[0].x, lr.nodes[0].y) for lr in self.laterals] 
        dists = sp.spatial.distance.cdist(primary_pts, lr_pts)
        self.branch_pts = [(i // n_ints, (i % n_ints) / n_ints) for i in np.argmin(dists,0)]

    def set_depths(self):
        '''Set attribute depth for each Branch in self.laterals'''
        for i in range(len(self.laterals)):
            j, d = self.branch_pts[i]
            self.laterals[i].depth = ((1-d) * self.primary.nodes[j].length + 
                                  d * self.primary.nodes[j+1].length)
            
    def set_branch_angles(self):
        '''Set attribute branch_ang for each Branch in self.laterals.'''
        for i in range(len(self.laterals)):
            if len(self.laterals[i].nodes) >= 2:
                pi, _ = self.branch_pts[i]
                p1 = self.primary.nodes[pi]
                p2 = self.primary.nodes[pi+1]
                l1 = self.laterals[i].nodes[0]
                l2 = self.laterals[i].nodes[1]
                
                self.laterals[i].angle = l1.angle_to(l2) - p1.angle_to(p2)

    def __str__(self) -> None:
        s = ["PRIMARY:", str(self.primary), "LATERALS:"] + [str(lr) for lr in self.laterals]
        return "\n".join(s)


def splitrsml2root(file, pixel_size_mm=79/1000, verbose=False):
    """
    Given an RSML of a split root system converts to two Root objects and a dictionary
    version of the RSML.

    New:
        verbose : bool
            If True, prints diagnostics during hr correction. If False, silences them.
    """
    _p = print if verbose else (lambda *a, **k: None)  # local printer

    # convert xml to dict:
    with open(file, 'r', encoding='utf-8') as fileobj:
        rsml = fileobj.read()
    RSA_dict = xmltodict.parse(rsml)

    if  RSA_dict['rsml']['scene']['plant'][0]['root']['geometry']['polyline']['point'][0]['@coord_x'] < RSA_dict['rsml']['scene']['plant'][1]['root']['geometry']['polyline']['point'][0]['@coord_x']:
        left = RSA_dict['rsml']['scene']['plant'][0]
        right = RSA_dict['rsml']['scene']['plant'][1]
    else:
        right = RSA_dict['rsml']['scene']['plant'][0]
        left  = RSA_dict['rsml']['scene']['plant'][1]

    def get_root(side):
        primary_dict = side['root']['geometry']['polyline']['point']
        primary = Branch(primary_dict, pixel_size_mm)

        laterals = []
        if 'root' in side['root']:
            if isinstance(side['root']['root'], list):
                for lr in side['root']['root']:
                    lateral_dict = lr['geometry']['polyline']['point']
                    if isinstance(lateral_dict, dict):
                        lateral_dict = [lateral_dict]
                    laterals.append(Branch(lateral_dict, pixel_size_mm))
            else:
                lr = side['root']['root']
                lateral_dict = lr['geometry']['polyline']['point']
                if isinstance(lateral_dict, dict):
                    lateral_dict = [lateral_dict]
                laterals.append(Branch(lateral_dict, pixel_size_mm))

        return Root(primary, laterals)

    left_rsa  = get_root(left)
    right_rsa = get_root(right)

    def _fix_first_node_hr_if_needed(lr):
        """
        Correct hr at index 0 if hr[0]==0 but time[0]!=0; return True if changed.
        """
        hr_values   = [node.hr  for node in lr.nodes]
        time_values = [node.time for node in lr.nodes]

        if len(hr_values) > 1 and len(time_values) > 1:
            if hr_values[0] == 0 and time_values[0] != 0:
                _p(f"FOUND: First node has hr=0 but time={time_values[0]}")
                _p(f"Original hr values: {hr_values}")
                _p(f"Time values: {time_values}")

                hr_values_0 = hr_values[1] - (time_values[1] - time_values[0])
                _p(f"Calculated hr_values_0: {hr_values_0}")

                lr.nodes[0].hr = hr_values_0

                updated_hr_values = [node.hr for node in lr.nodes]
                _p(f"Updated hr values: {updated_hr_values}")
                _p("-" * 50)
                return True
        return False

    def _apply_hr_fixes(root):
        if root.laterals:
            for lr in root.laterals:
                changed = _fix_first_node_hr_if_needed(lr)
                if changed:
                    # re-derive attributes from corrected hr
                    lr.set_data()

    _apply_hr_fixes(left_rsa)
    _apply_hr_fixes(right_rsa)

    return left_rsa, right_rsa, RSA_dict



if __name__ == "__main__":
    test_branch = Branch([Node(0, 4*i,i) for i in range(5)], 1)
    test_branches = [Branch([Node(1 + i, 3 * j,i + j) for i in range(5)]) for j in range(1,4)]
    test_root = Root(test_branch, test_branches)
    print(*test_root.branch_pts)
    print(*[lr.depth for lr in test_root.laterals])
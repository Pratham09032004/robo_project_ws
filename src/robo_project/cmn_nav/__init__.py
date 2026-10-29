"""Coarse Map Navigator (CMN, Xu et al., ICRA 2024) on ROS 2 + Habitat.

local_occ_model    - panoramic RGB -> egocentric local occupancy network (paper IV-B)
local_grid         - egocentric <-> map-aligned grid geometry shared by all parts
bayes_filter       - discrete Bayes filter on the coarse map (paper IV-C)
generate_dataset   - renders training data from Habitat (run in the conda habitat env)
train              - trains the local occupancy predictor
local_occ_node     - ROS 2 node: 4 cameras -> /cmn/local_occ
localization_node  - ROS 2 node: /cmn/local_occ + /odom deltas + /map -> /cmn/odom
"""

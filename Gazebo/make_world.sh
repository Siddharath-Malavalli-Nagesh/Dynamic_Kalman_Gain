#!/usr/bin/env bash
# Generate a custom Gazebo world with a specified gravity_z value.
# Usage:  bash make_world.sh -8.5  > custom.world
#
# Produces a minimal world that includes the standard ground plane,
# sun, and physics block with custom gravity. Compatible with the
# turtlebot3_gazebo empty_world launch (which accepts world:=<path>).
set -euo pipefail

GZ="${1:--9.81}"

cat <<EOF
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="default">
    <physics name="default_physics" default="0" type="ode">
      <gravity>0 0 ${GZ}</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
  </world>
</sdf>
EOF

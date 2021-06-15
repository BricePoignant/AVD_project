#!/usr/bin/env python3
import numpy as np
import math
from postprocessing import decode_netout

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_TRAFFICLIGHT = 1
STAY_STOPPED_TL = 2
DANGEROUS = 3
THRESH_PEDE = 5.0

STOP_COUNTS = 3

DECEL_THRESHOLD = 15 # distanza minima da dove cominciare a rallentare dal semaforo (spazio di frenata in funzione della velocità attuale)
LAST_CHECK_DISTANCE=7 #meters
DELTA_ORIENTATION=15

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, model, desired_speed,indx_intersections):
        self._lookahead = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state = FOLLOW_LANE
        self._follow_lead_vehicle = False
        self._obstacle = False
        self._goal_state = [0.0, 0.0, 0.0]
        self._goal_index = 0
        self._stop_count = 0
        self._lookahead_collision_index = 0
        self._model = model
        self._previous_state = None
        self._stop_count = 0
        self._desired_speed = desired_speed
        self._STOP_THRESHOLD_TL = 0
        self._indx_intersections=indx_intersections
        self._tl_state_history=[]
        self._depth_history=[]


    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed, tl_depth, traffic_light_state):
        """Handles state transitions and computes the goal state.

        args:
            waypoints: current waypoints to track (global frame).
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states:
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the STAY_STOPPED state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations)
                              before moving from stop sign.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any stop lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the stop line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_stop_signs() function to
        # understand it.

        self._depth_history.append(tl_depth)
        self._tl_state_history.append(traffic_light_state)
        #yaw_degree = ego_state[2] * 180 / np.pi

        print(f" speed: {ego_state[3]} depth: {tl_depth}")


        if self._state == FOLLOW_LANE:
            if self._obstacle:
                self._previous_state = self._state
                self._state = DANGEROUS


                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                
                self._goal_state[2]=0
            else:
                print("FOLLOW_LANE")
                # First, find the closest index to the ego vehicle.
                closest_len, closest_index = get_closest_index(waypoints, ego_state)

                # Next, find the goal index that lies within the lookahead distance
                # along the waypoints.
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1

                #_,intersect_found = self.check_for_intersection(waypoints, closest_index, goal_index)
                #print("intersezione ",intersect_found)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]

                if len(self._depth_history)>=3:
                    if self._depth_history[-3:]<=[DECEL_THRESHOLD]*3 and tl_depth>=LAST_CHECK_DISTANCE and tl_depth!=1000:
                        self._goal_state=self.compute_tl_goal(ego_state,tl_depth,waypoints,goal_index,DELTA_ORIENTATION)
                        self._state = DECELERATE_TO_TRAFFICLIGHT

        elif self._state == DECELERATE_TO_TRAFFICLIGHT:

            print("DECELERATE TO TRAFFIC LIGHT")

            if self._obstacle:
                self._previous_state = self._state
                self._state = DANGEROUS

                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                self._goal_state[2]=0
            else:
                if tl_depth<=LAST_CHECK_DISTANCE:
                    if self._tl_state_history[-3:]==[1,1,1]:
                        self._goal_state[2] = 0
                        self._state = STAY_STOPPED_TL
                    elif self._tl_state_history[-3:]==[0,0,0]:
                        self._state = FOLLOW_LANE

        elif self._state == STAY_STOPPED_TL:
            print("STAY STOPPED TL")
            if self._obstacle:

                self._previous_state = self._state
                self._state = DANGEROUS

                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                self._goal_state[2]=0
            else:

                if self._tl_state_history[-3:]==[0,0,0]:
                    closest_len, closest_index = get_closest_index(waypoints, ego_state)
                    goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                    while waypoints[goal_index][2] <= 0.1: goal_index += 1
                    self._goal_index = goal_index
                    self._goal_state = waypoints[goal_index]
                    self._state = FOLLOW_LANE

        elif self._state == DANGEROUS:
            print("DANGEROUS")

            if self._stop_count == STOP_COUNTS:
                if self._previous_state==FOLLOW_LANE:
                    closest_len, closest_index = get_closest_index(waypoints, ego_state)
                    goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                    self._goal_index = goal_index
                    self._goal_state = waypoints[goal_index]
                self._state = self._previous_state
                self._stop_count = 0

            # Otherwise, continue counting.
            else:
                self._stop_count += 1

        else:
            raise ValueError('Invalid state value.')



    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def compute_tl_goal(self, ego_state,tl_depth,waypoints,goal_index,DELTA_ORIENTATION=DELTA_ORIENTATION):
        yaw = ego_state[2] * 180 / np.pi
        new_vel=0.0
        if yaw >= 180 - DELTA_ORIENTATION or (yaw <= -180 + DELTA_ORIENTATION and yaw >= -180):
            new_x = ego_state[0] - (tl_depth-1)
            new_y = ego_state[1]

        elif yaw <= -90 + DELTA_ORIENTATION and yaw >= -90 - DELTA_ORIENTATION:
            new_x = ego_state[0]
            new_y = ego_state[1] - (tl_depth-1)

        elif yaw >= -DELTA_ORIENTATION and yaw <= DELTA_ORIENTATION:
            new_x = ego_state[0] + (tl_depth-1)
            new_y = ego_state[1]

        elif yaw >= 90 - DELTA_ORIENTATION and yaw <= 90 + DELTA_ORIENTATION:
            new_x = ego_state[0]
            new_y = ego_state[1] + (tl_depth-1)

        else:
            new_x = waypoints[goal_index][0]
            new_y = waypoints[goal_index][1]
            new_vel = waypoints[goal_index][2]
        return [new_x, new_y, new_vel]

    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle.

        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index + 1][0]) ** 2 + (
                        waypoints[wp_index][1] - waypoints[wp_index + 1][1]) ** 2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)

    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.

    def check_for_lead_vehicle(self, ego_state, lead_car_position):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.
            if lead_car_distance > self._follow_lead_vehicle_lookahead:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector,
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]),
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector,
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < self._follow_lead_vehicle_lookahead + 15:
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False

    def check_for_intersection(self, waypoints, closest_index, goal_index):

        #print("closest_index ",closest_index)
        #print("goal_index ", goal_index)

        for i in range(closest_index, goal_index):
            for indx in self._indx_intersections:
                if i == indx:
                    if i>=2 or True:
                        #return i-2, True
                        self._indx_intersections.remove(i)
                        return i, True
        return goal_index, False


# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]:
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0]) ** 2 + (waypoints[i][1] - ego_state[1]) ** 2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
            (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False

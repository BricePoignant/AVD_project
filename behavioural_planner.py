#!/usr/bin/env python3
import numpy as np
import math
import copy
import sys,os
sys.path.append(os.path.abspath(sys.path[0] + '/..'))


STOP_COUNTS = 3 #number of iterations to exit DANGEROUS state
DECEL_THRESHOLD = 15  #minimum distance in meters to enter in TRAFFICLIGHT_STOP state if a traffic light is detected
LAST_CHECK_DISTANCE=7  #minimum distance in meters to decide behaviour of the ego_vehicle close to traffic light
THRESHOLD_ORIENTATION=15 #used offset to check the ego_vehicle's orientation


# State machine states
FOLLOW_LANE = 0
TRAFFICLIGHT_STOP = 1
DANGEROUS = 2


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state = FOLLOW_LANE
        self._follow_lead_vehicle = False
        self._obstacle = False
        self._goal_state = [0.0, 0.0, 0.0]
        self._goal_index = 0
        self._stop_count = 0
        self._lookahead_collision_index = 0
        self._previous_state = -1
        self._stop_count = 0
        self._tl_state_history=[]
        self._depth_history=[]
        self._handbrake=False
        self._in_intersection=False


    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state,tl_depth, traffic_light_state):
        """Handles state transitions and computes the goal state.

        """
        print(f"BEHAVIOURAL PLANNER -> stato : {self._state} | precedente : {self._previous_state} | depth : {tl_depth}")
        self._depth_history.append(tl_depth)
        self._tl_state_history.append(traffic_light_state)
        self._handbrake=False
        self._in_intersection=False


        if self._follow_lead_vehicle:
            self._state = FOLLOW_LANE

        if self._state == FOLLOW_LANE:
            #check if there is a pedestrian close by and enter in DANGEROUS state
            if self._obstacle:
                self._previous_state = self._state
                self._state = DANGEROUS
                self._follow_lead_vehicle=False
                self._previous_goal_state = copy.deepcopy(self._goal_state)

            else:
                # First, find the closest index to the ego vehicle.
                closest_len, closest_index = get_closest_index(waypoints, ego_state)

                # Next, find the goal index that lies within the lookahead distance
                # along the waypoints.
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]

                #set the previous_state to FOLLOW_LANE if there are five consecutive no detections
                if self._tl_state_history[-5:]==[2,2,2,2,2]:
                    self._previous_state=FOLLOW_LANE

                #if the previous_state is not equal to TRAFFICLIGHT_STOP, then check if there is a traffic light closer the ego_vehicle
                if self._previous_state != TRAFFICLIGHT_STOP:
                    if len(self._depth_history)>=3:
                        depth_flag=False
                        for d in self._depth_history[-3:]:
                            if d<=DECEL_THRESHOLD:
                                depth_flag=True
                        #compute the next goal_state according to the position of traffic light if there is one
                        if depth_flag:
                            self._goal_state=self.compute_tl_goal(ego_state,tl_depth,waypoints,goal_index,THRESHOLD_ORIENTATION)
                            self._previous_state = self._state
                            self._state = TRAFFICLIGHT_STOP


        elif self._state == TRAFFICLIGHT_STOP:
            #check if there is a pedestrian close by and enter in DANGEROUS state
            if self._obstacle:
                self._previous_state = self._state
                self._state = DANGEROUS
                self._follow_lead_vehicle = False
                self._previous_goal_state=copy.deepcopy(self._goal_state)

            else:
                #decide  the ego_vehicle 's behaviour close to the traffic light
                if tl_depth<=LAST_CHECK_DISTANCE:
                    #decide the ego_vehicle 's behaviour if there are two consecutive traffic light red states
                    if self._tl_state_history[-2:]==[1,1]:
                        self._goal_state[2] = 0
                        if tl_depth<=4:
                            self._handbrake = True
                    #decide the ego_vehicle 's behaviour if there are six consecutive traffic light green states
                    elif self._tl_state_history[-6:]==[0,0,0,0,0,0]:
                        self._previous_state = self._state
                        self._state = FOLLOW_LANE
                #if we lose the traffic light detection and the ego_vehicle's speed is near 0, just move a bit
                elif self._tl_state_history[-5:]==[2,2,2,2,2] and ego_state[3]<=0.5:
                    self._goal_state[2]=1.5
                else:
                    #set the state to FOLLOW_LANE if there are seven consecutive no detections
                    if self._tl_state_history[-7:]==[2,2,2,2,2,2,2]:
                        self._state = FOLLOW_LANE

        elif self._state == DANGEROUS:
            #decrease the ego_vehicle' speed
            if  ego_state[3] >= 2.0:
                if self._goal_state[2] <= 0:
                    self._goal_state[2] = 0
                else:
                    self._goal_state[2] = 1.5


            if self._stop_count == STOP_COUNTS:
                #check If if there are no pedestrians
                if not self._obstacle:
                    #if the previous state is Follow lane state and the last computed depth is less than LAST_CHECK_DISTANCE then
                    #set the state to TRAFFICLIGHT_STOP
                    if self._previous_state==FOLLOW_LANE:
                        self._state = self._previous_state
                        self._goal_state = self._previous_goal_state
                        if self._depth_history[-1]<=LAST_CHECK_DISTANCE:
                            self._state=TRAFFICLIGHT_STOP
                            closest_len, closest_index = get_closest_index(waypoints, ego_state)
                            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                            #recompute goal_state
                            self._goal_state = self.compute_tl_goal(ego_state, tl_depth, waypoints, goal_index,THRESHOLD_ORIENTATION)
                    #if if the previous state is TRAFFICLIGHT_STOP state, go back in it and recover the previous goal state
                    elif self._previous_state==TRAFFICLIGHT_STOP:
                        self._state = self._previous_state
                        self._goal_state=self._previous_goal_state

                self._stop_count = 0

            # Otherwise, continue counting.
            else:
                self._stop_count += 1

        else:
            raise ValueError('Invalid state value.')



    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def compute_tl_goal(self, ego_state,tl_depth,waypoints,goal_index,THRESHOLD_ORIENTATION=THRESHOLD_ORIENTATION):
        '''compute the new goal_state according with traffic light position with respect to the camera position and set goal_speed to 2.5'''
        yaw = ego_state[2] * 180 / np.pi
        new_vel=2.5
        if yaw >= 180 - THRESHOLD_ORIENTATION or (yaw <= -180 + THRESHOLD_ORIENTATION and yaw >= -180):
            new_x = ego_state[0] - (tl_depth)
            new_y = ego_state[1]

        elif yaw <= -90 + THRESHOLD_ORIENTATION and yaw >= -90 - THRESHOLD_ORIENTATION:
            new_x = ego_state[0]
            new_y = ego_state[1] - (tl_depth)

        elif yaw >= -THRESHOLD_ORIENTATION and yaw <= THRESHOLD_ORIENTATION:
            new_x = ego_state[0] + (tl_depth)
            new_y = ego_state[1]

        elif yaw >= 90 - THRESHOLD_ORIENTATION and yaw <= 90 + THRESHOLD_ORIENTATION:
            new_x = ego_state[0]
            new_y = ego_state[1] + (tl_depth)

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

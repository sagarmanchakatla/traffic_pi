from typing import Dict, List, Any
from pydantic import BaseModel

class LaneTiming(BaseModel):
    green: int
    yellow: int

class TrafficTimingOptimizer:
    def __init__(self):
        # Saturation flows (PCU - Passenger Car Units)
        self.saturation_flows = {
            'car': 1.0, 
            'motorcycle': 0.5, 
            'bus': 2.0, 
            'truck': 2.5
        }
        
        # Timing constants
        self.YELLOW_TIME = 4
        self.ALL_RED_TIME = 2
        self.MIN_GREEN = 10
        self.MAX_GREEN = 60
        self.CYCLE_MAX = 180

    def calculate_effective_demand(self, counts: Dict[str, int]) -> float:
        """
        Calculate effective demand (PCU) for a lane based on vehicle counts.
        """
        demand = 0.0
        for vehicle_type, count in counts.items():
            # Default to 1.0 if type not found (e.g. 'person' or unknown)
            factor = self.saturation_flows.get(vehicle_type, 1.0)
            demand += count * factor
        return demand

    def optimize(self, lane_counts: Dict[str, Dict[str, int]], lane_config: Dict[str, Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Calculate optimal timings and priority based on vehicle counts and lane configuration.
        
        lane_config format:
        {
            "lane1": {"hasLeft": True, "hasRight": False},
            ...
        }
        """
        
        effective_demands = {}
        total_demand = 0.0
        
        # 1. Calculate effective demand per lane
        for lane, counts in lane_counts.items():
            demand = self.calculate_effective_demand(counts)
            effective_demands[lane] = demand
            total_demand += demand
            
        # 2. Determine Priority (Sort lanes by demand descending)
        priority_order = sorted(effective_demands.keys(), key=lambda k: effective_demands[k], reverse=True)
        
        # 3. Calculate Cycle Time
        total_lost_time = (self.YELLOW_TIME + self.ALL_RED_TIME) * len(lane_counts)
        
        # Add lost time for left turns if present (extra phase transition)
        if lane_config:
            for lane, config in lane_config.items():
                if config.get("hasLeft"):
                    total_lost_time += (self.YELLOW_TIME) # Extra yellow for the left turn phase
        
        if total_demand == 0:
            # Fallback for zero traffic
            timings = {}
            for lane in lane_counts:
                timings[lane] = {
                    "green": 15, 
                    "yellow": self.YELLOW_TIME,
                    "leftGreen": 0,
                    "leftYellow": 0
                }
                
            return {
                "timings": timings,
                "priority": list(lane_counts.keys()),
                "totalTime": (15 + self.YELLOW_TIME + self.ALL_RED_TIME) * 4,
                "effectiveDemands": effective_demands
            }

        # Dynamic cycle length
        calculated_cycle = (total_demand * 2.5) + total_lost_time
        cycle_time = max(60, min(calculated_cycle, self.CYCLE_MAX))
        
        available_green_time = cycle_time - total_lost_time
        
        # 4. Allocate Green Time
        timings = {}
        for lane in lane_counts:
            if total_demand > 0:
                ratio = effective_demands[lane] / total_demand
                total_lane_green = int(ratio * available_green_time)
            else:
                total_lane_green = self.MIN_GREEN
                
            # Clamp total green time
            total_lane_green = max(self.MIN_GREEN, min(total_lane_green, self.MAX_GREEN))
            
            # Split into Main and Left if applicable
            has_left = lane_config and lane_config.get(lane, {}).get("hasLeft", False)
            
            if has_left:
                # Allocate ~30% to left turn, but at least 5s if total is enough
                left_green = max(5, int(total_lane_green * 0.3))
                main_green = max(self.MIN_GREEN, total_lane_green - left_green)
                
                timings[lane] = {
                    "green": main_green,
                    "yellow": self.YELLOW_TIME,
                    "leftGreen": left_green,
                    "leftYellow": self.YELLOW_TIME # Transition from Left -> Main usually doesn't need yellow if concurrent, but for Leading Left -> Main (Straight), we might just switch. 
                    # Let's assume Leading Left: Left Green -> Left Yellow -> Main Green (Left Red).
                }
            else:
                timings[lane] = {
                    "green": total_lane_green,
                    "yellow": self.YELLOW_TIME,
                    "leftGreen": 0,
                    "leftYellow": 0
                }
            
        # Recalculate actual total time
        actual_total_time = 0
        for lane in priority_order:
            t = timings[lane]
            lane_time = t["green"] + t["yellow"] + self.ALL_RED_TIME
            if t["leftGreen"] > 0:
                lane_time += t["leftGreen"] + t["leftYellow"]
            actual_total_time += lane_time

        return {
            "timings": timings,
            "priority": priority_order,
            "totalTime": actual_total_time,
            "effectiveDemands": effective_demands
        }

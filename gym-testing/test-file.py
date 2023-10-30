from enum import Enum


class Actions(Enum):
   pass

for i in range(25):
    for j in range(17):
        if (j == 0):
            setattr(Actions, f"ACTION_{i+1}_{j}", "Turning ON")
        elif (j == 16):
            setattr(Actions, f"ACTION_{i+1}_{j}", "Do nothing")
        else:
            setattr(Actions, f"ACTION_{i+1}_{j}", "Turning OFF and shift load")
        
action = Actions.ACTION_1_0
print(action)
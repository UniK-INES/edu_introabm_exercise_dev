from os import listdir, path

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import FireEvacuation
from .agent import FireExit, Wall, Human, Sight, Door


# Creates a visual portrayal of our model in the browser interface
def fire_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {}
    (x, y) = agent.get_position()
    portrayal["x"] = x
    portrayal["y"] = y

    if type(agent) is Human:
        portrayal["scale"] = 1
        portrayal["Layer"] = 5

        if agent.get_mobility() == Human.Mobility.INCAPACITATED:
            # Incapacitated
            portrayal["Shape"] = "fire_evacuation/resources/incapacitated_human.png"
            portrayal["Layer"] = 6
        elif agent.get_mobility() == Human.Mobility.PANIC:
            # Panicked
            portrayal["Shape"] = "fire_evacuation/resources/panicked_human.png"
        else:
            # Normal
            portrayal["Shape"] = "fire_evacuation/resources/human.png"
    elif type(agent) is FireExit:
        portrayal["Shape"] = "fire_evacuation/resources/fire_exit.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Door:
        portrayal["Shape"] = "fire_evacuation/resources/door.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Wall:
        portrayal["Shape"] = "fire_evacuation/resources/wall.png"
        portrayal["scale"] = 1
        portrayal["Layer"] = 1
    elif type(agent) is Sight:
        portrayal["Shape"] = "fire_evacuation/resources/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 7

    return portrayal

# initial extent that will be changed on reset (?)
canvas_element = CanvasGrid(fire_evacuation_portrayal, 50, 50, 800, 800)

# Define the charts on our web interface visualisation
status_chart = ChartModule(
    [
        {"Label": "Alive", "Color": "blue"},
        {"Label": "Dead", "Color": "red"},
        {"Label": "Escaped", "Color": "green"},
    ]
)

mobility_chart = ChartModule(
    [
        {"Label": "Normal", "Color": "green"},
        {"Label": "Panic", "Color": "red"},
        {"Label": "Incapacitated", "Color": "blue"},
    ]
)



# Specify the parameters changeable by the user, in the web interface
model_params = {
    "floor_size": UserSettableParameter(
        "slider", "Room size (edge)", value=50, min_value=10, max_value=1000, step=10
    ),
    "human_count": UserSettableParameter(
        "slider", "Number Of Human Agents", value=10, min_value=1, max_value=10000, step=10),
    "random_spawn": UserSettableParameter(
        "checkbox", "Spawn Agents at Random Locations", value=True
    ),
    "visualise_vision": UserSettableParameter("checkbox", "Show Agent Vision", value=False),
    "save_plots": UserSettableParameter("checkbox", "Save plots to file", value=True),
}

# Start the visual server with the model
server = ModularServer(
    FireEvacuation,
    [canvas_element, status_chart, mobility_chart],
    "Room Evacuation",
    model_params,
)

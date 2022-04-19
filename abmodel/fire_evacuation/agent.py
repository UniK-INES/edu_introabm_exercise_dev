from typing import Union
from typing_extensions import Self
from mesa.space import Coordinate
import networkx as nx
import numpy as np
from enum import IntEnum
from mesa import Agent
from copy import deepcopy

from fire_evacuation.utils import get_random_id


def get_line(start, end):
    """
    Implementaiton of Bresenham's Line Algorithm
    Returns a list of tuple coordinates from starting tuple to end tuple (and including them)
    """
    # Break down start and end tuples
    x1, y1 = start
    x2, y2 = end

    # Calculate differences
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Check if the line is steep
    line_is_steep = abs(diff_y) > abs(diff_x)

    # If the line is steep, rotate it
    if line_is_steep:
        # Swap x and y values for each pair
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # If the start point is further along the x-axis than the end point, swap start and end
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Calculate the differences again
    diff_x = x2 - x1
    diff_y = y2 - y1

    # Calculate the error margin
    error_margin = int(diff_x / 2.0)
    step_y = 1 if y1 < y2 else -1

    # Iterate over the bounding box, generating coordinates between the start and end coordinates
    y = y1
    path = []

    for x in range(x1, x2 + 1):
        # Get a coordinate according to if x and y values were swapped
        coord = (y, x) if line_is_steep else (x, y)
        path.append(coord)  # Add it to our path
        # Deduct the absolute difference of y values from our error_margin
        error_margin -= abs(diff_y)

        # When the error margin drops below zero, increase y by the step and the error_margin by the x difference
        if error_margin < 0:
            y += step_y
            error_margin += diff_x

    # The the start and end were swapped, reverse the path
    if swapped:
        path.reverse()

    return path


"""
FLOOR STUFF
"""


class FloorObject(Agent):
    def __init__(
        self,
        pos: Coordinate,
        traversable: bool,
        visibility: int = 2,
        model=None,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)
        self.pos = pos
        self.traversable = traversable
        self.visibility = visibility

    def get_position(self):
        return self.pos


class Sight(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, visibility=-1, model=model
        )

    def get_position(self):
        return self.pos


class Door(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, model=model)


class FireExit(FloorObject):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, visibility=6, model=model)


class Wall(FloorObject):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, model=model)


class Furniture(FloorObject):
    
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, model=model)


class Human(Agent):
    """
    A human agent, which will attempt to escape from the grid.

    Attributes:
        ID: Unique identifier of the Agent
        Position (x,y): Position of the agent on the Grid
        Health: Health of the agent (between 0 and 1)
        ...
    """

    class Mobility(IntEnum):
        """"
        Agents' mobility
        """
        INCAPACITATED = 0
        NORMAL = 1
        PANIC = 2

    class Status(IntEnum):
        
        ALIVE = 1
        ESCAPED = 2
        

    MIN_HEALTH = 0.0
    MAX_HEALTH = 1.0

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_NERVOUSNESS = 0.1
    MAX_NERVOUSNESS = 1.0
    
    MIN_SPEED = 0.0
    MAX_SPEED = 2.0

    MIN_KNOWLEDGE = 0
    MAX_KNOWLEDGE = 1
    
    # The value the panic score must reach for an agent to start panic behaviour
    PANIC_THRESHOLD = 0.8
    DEFAULT_NERVOUSNESS_MODIFIER = -0.1
    NERVOUSNESS_MODIFIER_AFFECTED_HUMAN = 0.3
    
    # When the health value drops below this value, the agent will being to slow down
    SLOWDOWN_THRESHOLD = 0.5

    MIN_PUSH_DAMAGE = 0.01
    MAX_PUSH_DAMAGE = 0.2

        
    def __init__(self,
            pos: Coordinate,
            health: float,
            speed: float,
            vision: int,
            nervousness,
            experience,
            believes_alarm: bool,
            model,
        ):
        
        rand_id = get_random_id()
        super().__init__(rand_id, model)

        ''' Human agents should not be traversable, but we allow 
        "displacement", e.g. pushing to the side'''
        self.traversable = False

        self.pos = pos
        self.visibility = 2
        self.health = health
        self.mobility: Human.Mobility = Human.Mobility.NORMAL
        self.speed = speed
        self.vision = vision

        self.knowledge = self.MIN_KNOWLEDGE
        self.nervousness = nervousness
        self.experience = experience
        # Boolean stating whether or not the agent believes the alarm is a real fire
        self.believes_alarm = believes_alarm  
        self.escaped: bool = False

        # The agent and seen location (agent, (x, y)) the agent is planning to move to
        self.planned_target: tuple[Agent, Coordinate] = (
            None,
            None,
        )

        self.visible_tiles: tuple[Coordinate, tuple[Agent]] = []

        # An initially empty set representing what the agent knows of the floor plan
        self.known_tiles: dict[Coordinate, set[Agent]] = {}

        self.visited_tiles: set[Coordinate] = {self.pos}
        
        ''' An action the agent intends to do when they reach their 
        planned target {"carry", "morale"}'''
        self.planned_action: Human.Action = None  


    def update_sight_tiles(self, visible_neighborhood):
        """
        Update visible tiles

        Parameters
        ----------
        visible_neighborhood : set
            Set of visible tile and their visible content in neighborhood

        Returns
        -------
        None.

        """
        if len(self.visible_tiles) > 0:
            # Remove old vision tiles
            for pos, _ in self.visible_tiles:
                contents = self.model.grid.get_cell_list_contents(pos)
                for agent in contents:
                    if isinstance(agent, Sight):
                        self.model.grid.remove_agent(agent)

        # Add new vision tiles
        for contents, tile in visible_neighborhood:
            # Only place empty tiles or those with visible content
            if self.model.grid.is_cell_empty(tile) or len(contents) > 0:
                sight_object = Sight(tile, self.model)
                self.model.grid.place_agent(sight_object, tile)
 
    def get_visible_tiles(self) -> tuple[Coordinate, tuple[Agent]]:
        """
        A strange implementation of ray-casting, using Bresenham's Line Algorithm,
        which takes into account visibility of objects

        Returns
        -------
        tuple[Coordinate, tuple[Agent]]
            Tiles and their visible content.

        """
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision
        )
        visible_neighborhood = set()

        # A set of already checked tiles
        checked_tiles = set()

        # Reverse the neighborhood so we start from the furthest locations and work our way inwards
        for pos in reversed(neighborhood):
            if pos not in checked_tiles:
                blocked = False
                try:
                    path = get_line(self.pos, pos)

                    for i, tile in enumerate(path):
                        contents = self.model.grid.get_cell_list_contents(tile)
                        visible_contents = []
                        for obj in contents:
                            if isinstance(obj, Sight):
                                # ignore sight tiles
                                continue
                            elif isinstance(obj, Wall):
                                # We hit a wall, reject rest of path and move to next
                                blocked = True
                                break
                  
                            # If the object has a visibility score it is visible
                            if obj.visibility:
                                visible_contents.append(obj)

                        if blocked:
                            '''Add the rest of the path to checked tiles, 
                            since we now know they are not visible'''
                            checked_tiles.update(
                                path[i:]
                            )  
                            break
                        else:
                            # If a wall didn't block the way, add the visible 
                            # agents at this location
                            checked_tiles.add(
                                tile
                            )
                            # Add the tile to checked tiles so we don't check it again
                            visible_neighborhood.add((tile, tuple(visible_contents)))

                except Exception as e:
                    print(e)

        if self.model.visualise_vision:
            self.update_sight_tiles(visible_neighborhood)

        return tuple(visible_neighborhood)

    def get_random_target(self, allow_visited=True):
        """
        Choose random tile from known, traversable tiles

        Parameters
        ----------
        allow_visited : bool, optional
            The default is True.

        Returns
        -------
        None.

        """
        graph_nodes = self.model.graph.nodes()

        known_pos = set(self.known_tiles.keys())

        # If we are excluding visited tiles, remove the visited_tiles set 
        # from the available tiles
        if not allow_visited:
            known_pos -= self.visited_tiles

        traversable_pos = [pos for pos in known_pos if self.location_is_traversable(pos)]

        while not self.planned_target[1] and len(traversable_pos) > 0:
            i = np.random.choice(len(traversable_pos))
            target_pos = traversable_pos[i]
            if target_pos in graph_nodes and target_pos != self.pos:
                # No agent as target
                self.planned_target = (None, target_pos)

    def attempt_exit_plan(self):
        """
        Finds a target to exit

        Returns
        -------
        None.

        """
        self.planned_target = (None, None)
        fire_exits = set()

        # Query fire exits from known tiles
        for pos, agents in self.known_tiles.items():
            for agent in agents:
                if isinstance(agent, FireExit):
                    fire_exits.add((agent, pos))

        if len(fire_exits) > 0:
            if len(fire_exits) > 1:  
                # If there is more than one exit known
                best_distance = None
                for exit, exit_pos in fire_exits:
                    # Let's use Bresenham's to find the 'closest' exit
                    length = len(get_line(self.pos, exit_pos))
                    if not best_distance or length < best_distance:
                        best_distance = length
                        self.planned_target = (exit, exit_pos)

            else:
                self.planned_target = fire_exits.pop()

        else:
            # If there's no fire-escape in sight, try to head for an
            # unvisited door - no shortest way optimisation
            # (assumption: only one door per room)
            found_door = False
            for pos, contents in self.visible_tiles:
                for agent in contents:
                    if isinstance(agent, Door):
                        found_door = True
                        self.planned_target = (agent, pos)
                        break

                if found_door:
                    break

            # Still didn't find a planned_target, so get a random unvisited target
            if not self.planned_target[1]:
                self.get_random_target(allow_visited=False)

    def get_panic_score(self):
        """
        Calculate panic score

        Returns
        -------
        panic_score : TYPE

        """
        health_component = 1 / np.exp(self.health / self.nervousness)
        experience_component = 1 / np.exp(self.experience / self.nervousness)

        # Calculate the mean of the components
        panic_score = (health_component + experience_component) / 2
        return panic_score
    

    def incapacitate(self):
        """
        Called when agents become incapacitated

        Returns
        -------
        None.

        """
        self.mobility = Human.Mobility.INCAPACITATED
        self.traversable = True


    def health_speed_rules(self):
        """
        Adapt agent's healt and speed

        Returns
        -------
        None.

        """
        # Start to slow the agent when they drop below 50% health
        if self.health < self.SLOWDOWN_THRESHOLD:
            self.speed -= self.SPEED_MODIFIER_SMOKE

        # Prevent health and speed from going below 0
        if self.speed < self.MIN_SPEED:
            self.speed = self.MIN_SPEED

        elif self.speed == self.MIN_SPEED:
            self.incapacitate()

    def panic_rules(self):
        """
        Apply panic rules (when nervousness exceeds self.PANIC_THRESHOLD)

        Returns
        -------
        None.

        """
        # Shock will decrease by this amount if no new shock is added
        nervousness_modifier = self.DEFAULT_NERVOUSNESS_MODIFIER
        for _, agents in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Human) and agent.get_mobility() != Human.Mobility.NORMAL:
                    nervousness_modifier += (
                        self.NERVOUSNESS_MODIFIER_AFFECTED_HUMAN - self.DEFAULT_NERVOUSNESS_MODIFIER
                    )

        # If the agent's shock value increased and they didn't believe the alarm before, 
        # they now do believe it
        if not self.believes_alarm and nervousness_modifier != self.DEFAULT_NERVOUSNESS_MODIFIER:
            self.believes_alarm = True

        self.nervousness += nervousness_modifier

        # Keep the nervousness value between 0 and 1
        if self.nervousness > self.MAX_NERVOUSNESS:
            self.nervousness = self.MAX_NERVOUSNESS
        elif self.nervousness < self.MIN_NERVOUSNESS:
            self.nervousness = self.MIN_NERVOUSNESS

        panic_score = self.get_panic_score()

        if panic_score >= self.PANIC_THRESHOLD:
            self.mobility = Human.Mobility.PANIC

            # when an agent panics, clear known tiles
            # this represents the agent forgetting all logical information about their surroundings,
            # and having to rebuild it once they stop panicking
            self.known_tiles = {}
            self.knowledge = 0
        elif panic_score < self.PANIC_THRESHOLD and self.mobility == Human.Mobility.PANIC:
            self.mobility = Human.Mobility.NORMAL


    def learn_environment(self):
        """
        Learn content of visible tiles

        Returns
        -------
        None.

        """
        if self.knowledge < self.MAX_KNOWLEDGE:  # If there is still something to learn
            new_tiles = 0

            for pos, agents in self.visible_tiles:
                if pos not in self.known_tiles.keys():
                    new_tiles += 1
                self.known_tiles[pos] = set(agents)

            # update the knowledge Attribute accordingly
            total_tiles = self.model.grid.width * self.model.grid.height
            new_knowledge_percentage = new_tiles / total_tiles
            self.knowledge = self.knowledge + new_knowledge_percentage
            

    def get_next_location(self, path):
        """
        Extract the path and target for the next tick.

        Parameters
        ----------
        path : tuple
            currently followed path.

        Raises
        ------
        Exception
            Failure when determiniing next location.

        Returns
        -------
        next_location : pos
            Next location to end at.
        next_path : tuple
            Path to next location.

        """
        path_length = len(path)
        speed_int = int(np.round(self.speed))

        try:
            if path_length <= speed_int:
                next_location = path[path_length - 1]
            else:
                next_location = path[speed_int]

            next_path = []
            for location in path:
                next_path.append(location)
                if location == next_location:
                    break

            return (next_location, next_path)
        except Exception as e:
            raise Exception(
                f"Failed to get next location: {e}\nPath: {path},\nlen: {path_length},\nSpeed: {self.speed}"
            )

    def get_path(self, graph, target, include_target=True) -> list[Coordinate]:
        """
        Get path to target from graph

        Parameters
        ----------
        graph : nx graph
            graph of traversable ways over the floor plan.
        target : tile
            target tile
        include_target : bool, optional
            The default is True.

        Raises
        ------
        Exception
            DESCRIPTION.
        e
            DESCRIPTION.

        Returns
        -------
        list[Coordinate]
            an empty path if no path can be found

        """
        path = []
        visible_tiles_pos = [pos for pos, _ in self.visible_tiles]

        try:
            if target in visible_tiles_pos:  # Target is visible, so simply take the shortest path
                path = nx.shortest_path(graph, self.pos, target)
            else:  # Target is not visible, so do less efficient pathing
                # TODO: In the future this could be replaced with a more naive path algorithm
                # TODO check peformance
                path = nx.shortest_path(graph, self.pos, target)

                if not include_target:
                    del path[
                        -1
                    ]  # We don't want the target included in the path, so delete the last element

            return list(path)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if target not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(target)
                print(f"Target node not found! Expected {target}, with contents {contents}")
                return path
            elif self.pos not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(self.pos)
                raise Exception(
                    f"Current position not found!\nPosition: {self.pos},\nContents: {contents}"
                )
            else:
                raise e

        except nx.exception.NetworkXNoPath as e:
            print(f"No path between nodes! ({self.pos} -> {target})")
            return path


    def location_is_traversable(self, pos) -> bool:
        if not self.model.grid.is_cell_empty(pos):
            contents = self.model.grid.get_cell_list_contents(pos)
            for agent in contents:
                if not agent.traversable:
                    return False

        return True


    def push_human_agent(self, agent: Self):
        """
        Pushes the agent to a neighbouring tile

        Parameters
        ----------
        agent : Self
            agent to push.

        Returns
        -------
        None.

        """
        neighborhood = self.model.grid.get_neighborhood(
            agent.get_position(),
            moore=True,
            include_center=False,
            radius=1,
        )
        traversable_neighborhood = [
            neighbor_pos
            for neighbor_pos in neighborhood
            if self.location_is_traversable(neighbor_pos)
        ]

        if len(traversable_neighborhood) > 0:
            # push the human agent to a random traversable position
            i = np.random.choice(len(traversable_neighborhood))
            push_pos = traversable_neighborhood[i]
            self.model.grid.move_agent(agent, push_pos)

            # inure the pushed agent slightly
            current_health = agent.get_health()
            damage = np.random.uniform(self.MIN_PUSH_DAMAGE, self.MAX_PUSH_DAMAGE)
            agent.set_health(current_health - damage)


    def move_toward_target(self):
        next_location: Coordinate = None
        pruned_edges = set()
        # TODO inefficient?
        #graph = deepcopy(self.model.graph)
        graph = self.model.graph

        while self.planned_target[1] and not next_location:
            path = self.get_path(graph, self.planned_target[1])

            if len(path) > 0:
                next_location, next_path = self.get_next_location(path)

                if next_location == self.pos:
                    continue

                if next_location == None:
                    raise Exception("Next location can't be none")

                # Test the next location to see if we can move there
                if self.location_is_traversable(next_location):
                    # Move normally
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    self.visited_tiles.add(next_location)

                elif self.pos == path[-1]:
                    # The human reached their target!

                    if self.planned_action:
                        self.perform_action()

                    self.planned_target = (None, None)
                    self.planned_action = None
                    break

                else:
                    # We want to move here but it's blocked

                    # check if the location is blocked due to a Human agent
                    pushed = False
                    contents = self.model.grid.get_cell_list_contents(next_location)
                    for agent in contents:
                        # Test the panic value to see if this agent "pushes" the 
                        # blocking agent aside
                        if (
                            isinstance(agent, Human)
                            and agent.mobility != Human.Mobility.INCAPACITATED
                        ) and (
                            (
                                self.get_panic_score() >= self.PANIC_THRESHOLD
                                and self.mobility == Human.Mobility.NORMAL
                            )
                            or self.mobility == Human.Mobility.PANIC
                        ):
                            # push the agent and then move to the next_location
                            self.push_human_agent(agent)
                            self.previous_pos = self.pos
                            self.model.grid.move_agent(self, next_location)
                            self.visited_tiles.add(next_location)
                            pushed = True
                            break
                    if pushed:
                        continue

                    # Remove the next location from the temporary graph so we 
                    # can try pathing again without it
                    edges = graph.edges(next_location)
                    pruned_edges.update(edges)
                    graph.remove_node(next_location)

                    # Reset planned_target if the next location was the end of the path
                    if next_location == path[-1]:
                        next_location = None
                        self.planned_target = (None, None)
                        self.planned_action = None
                        break
                    else:
                        next_location = None

            else:  # No path is possible, so drop the target
                self.planned_target = (None, None)
                self.planned_action = None
                break

        if len(pruned_edges) > 0:
            # TODO does not seem to be necessary, as graph is not used after this in this function
            # Add back the edges we removed when removing any non-traversable nodes 
            # from the global graph, because they may be traversable again next step
            graph.add_edges_from(list(pruned_edges))

    def step(self):
        if not self.escaped and self.pos:
            self.health_speed_rules()

            if self.mobility == Human.Mobility.INCAPACITATED or not self.pos:
                # Incapacitated or died, so return already
                return

            self.visible_tiles = self.get_visible_tiles()

            self.panic_rules()

            self.learn_environment()

            planned_target_agent = self.planned_target[0]

            # If a fire has started and the agent believes it, attempt to plan 
            # an exit location if we haven't already and we aren't performing an action
            if self.believes_alarm:
                if not isinstance(planned_target_agent, FireExit) and not self.planned_action:
                    self.attempt_exit_plan()

            planned_pos = self.planned_target[1]
            if not planned_pos:
                self.get_random_target()
            elif self.mobility == Human.Mobility.PANIC:  # Panic
                panic_score = self.get_panic_score()

                if panic_score > 0.9 and np.random.random() < panic_score:
                    # If they have above 90% panic score, test the score to see if they faint
                    self.incapacitate()
                    return

            self.move_toward_target()

            # Agent reached a fire escape, proceed to exit
            if self.pos in self.model.fire_exits.keys():
                self.escaped = True
                self.model.grid.remove_agent(self)

    def get_status(self):
        if self.health > self.MIN_HEALTH and not self.escaped:
            return Human.Status.ALIVE
        elif self.escaped:
            return Human.Status.ESCAPED

        return None

    def get_speed(self):
        return self.speed

    def get_mobility(self):
        return self.mobility

    def get_health(self):
        return self.health

    def get_position(self):
        return self.pos

    def get_plan(self):
        return (self.planned_target, self.planned_action)

    def set_plan(self, agent, location):
        self.planned_action = None
        self.planned_target = (agent, location)

    def set_health(self, value: float):
        self.health = value

    def set_believes(self, value: bool):
        if value and not self.believes_alarm:
            self.believes_alarm = value
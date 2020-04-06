import sys
import os
import numpy as np
np.random.seed(11)
import osmnx as ox
import networkx as nx
import operator
from geopy.distance import vincenty,distance
from geopy import distance
import folium
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

address = ['דב האוזנר 3, תל אביב יפו', 'איזק שטרן 11, תל אביב', 'שדרות לוי אשכול 54, תל אביב']

# Translate address(written in English\Hebrew) to lat,lon cordinates.
def addressToLocations(address):
    locations = []
    locator = Nominatim(user_agent="myGeocoder")

    for addres in address:
        loc = locator.geocode(addres)
        location = (loc.latitude, loc.longitude)
        locations.append(location)

    return locations


# Calculating the distances array betwenen all pairs of locations using NetworKX and OSMnx libraries
def calculate_distances_array(locations):
    n = len(locations)
    distances_array = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                distances_array[i][j] = 0
            elif i != j and distances_array[i][j] == 0:
                l = tuple(map(operator.add, locations[i], locations[j]))
                l = (l[0]/2,l[1]/2)

                # Great circle distance between two points
                # vincenity: lat,lon
                dist = distance.distance((locations[i][0], locations[i][1]),
                    (locations[j][0], locations[j][1])).meters + 300

                graph = ox.graph_from_point(l,dist)

                # Finfing the nearest node to each one of the locations
                orig_node = ox.get_nearest_node(graph, locations[i])
                dest_node = ox.get_nearest_node(graph, locations[j])

                # Finding the shortest path between location i to j and from j to i (with metric = length)
                route_ij = nx.shortest_path_length(graph, source=orig_node, target=dest_node, weight='length')
                route_ji = nx.shortest_path_length(graph, source=dest_node, target=orig_node, weight='length')

                distances_array[i][j] = route_ij
                distances_array[j][i] = route_ji
    return distances_array


def DP_TSP(distances_array):
    n = len(distances_array)
    all_points_set = set(range(n))


    memo = {(tuple([i]), i): tuple([0, None]) for i in range(n)}
    queue = [(tuple([i]), i) for i in range(n)]

    while queue:
        prev_visited, prev_last_point = queue.pop(0)
        prev_dist, _ = memo[(prev_visited, prev_last_point)]

        to_visit = all_points_set.difference(set(prev_visited))
        for new_last_point in to_visit:
            new_visited = tuple(sorted(list(prev_visited) + [new_last_point]))
            new_dist = prev_dist + distances_array[prev_last_point][new_last_point]

            if (new_visited, new_last_point) not in memo:
                memo[(new_visited, new_last_point)] = (new_dist, prev_last_point)
                queue += [(new_visited, new_last_point)]
            else:
                if new_dist < memo[(new_visited, new_last_point)][0]:
                    memo[(new_visited, new_last_point)] = (new_dist, prev_last_point)

    optimal_path, optimal_cost = retrace_optimal_path(memo, n)

    return optimal_path, optimal_cost

def retrace_optimal_path(memo: dict, n: int) -> [[int], float]:
    points_to_retrace = tuple(range(n))

    full_path_memo = dict((k, v) for k, v in memo.items() if k[0] == points_to_retrace)
    path_key = min(full_path_memo.keys(), key=lambda x: full_path_memo[x][0])

    last_point = path_key[1]
    optimal_cost, next_to_last_point = memo[path_key]

    optimal_path = [last_point]
    points_to_retrace = tuple(sorted(set(points_to_retrace).difference({last_point})))

    while next_to_last_point is not None:
        last_point = next_to_last_point
        path_key = (points_to_retrace, last_point)
        _, next_to_last_point = memo[path_key]

        optimal_path = [last_point] + optimal_path
        points_to_retrace = tuple(sorted(set(points_to_retrace).difference({last_point})))

    return optimal_path, optimal_cost


def mergeNplot(optimal_path,locations):
    lat = lon = 0
    # Finding the center of all locations, it's purpose is for generating the map
    for location in locations:
        lat += location[0]
        lon += location[1]
    lat = lat/len(locations)
    lon = lon/len(locations)
    mid_point = (lat,lon)

    # Calculating the distace from each location to mid_point
    distances = [distance.distance(x,mid_point).meters for x in locations]
    dist = max(distances) + 200

    #Generating the graph
    graph = ox.graph_from_point(mid_point,dist)
    graph_map = ox.plot_graph_folium(graph, popup_attribute='name', edge_width=0)

    # Generating the route according to the optimal order of the locations
    routes = []
    for i in range(len(optimal_path)-1):
        l1 = locations[optimal_path[i]]
        l2 = locations[optimal_path[i+1]]

        orig_node = ox.get_nearest_node(graph, l1)
        dest_node = ox.get_nearest_node(graph, l2)
        route = nx.shortest_path(graph, source=orig_node, target=dest_node, weight='length')
        route_graph_map = ox.plot_route_folium(graph, route, route_map=graph_map)
        routes.append(route)

    # Markers addition on the places locations.
    i = 0
    for location in optimal_path:
        folium.Marker(location=locations[location],popup=i,icon=folium.Icon(color='blue')).add_to(graph_map)
        i+=1


    graph_map.save('tsp_graph.html')





locations = addressToLocations(address)
distances_array = calculate_distances_array(locations)
optimal_path, optimal_cost = DP_TSP(distances_array)
print(optimal_path)
mergeNplot(optimal_path,locations)


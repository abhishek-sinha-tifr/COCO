import heapq
import random
import numpy as np

# Djikstra's algorithm to find the shortest path
def djikstra(graph, start, gradient_F_t, edge_to_index):
    queue = [(0, start)]  # (distance, vertex)
    distances = {vertex: float('inf') for vertex in graph}  # Initialize distances to infinity
    distances[start] = 0  # Distance to the start vertex is 0
    previous_vertices = {vertex: None for vertex in graph}  # To reconstruct the shortest path
    visited = set()  # Set to track visited vertices

    while queue:
        current_distance, current_vertex = heapq.heappop(queue)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        # Update the distances to the neighboring vertices using edge weights from the matrix
        for neighbor in graph[current_vertex]:
            if neighbor in visited:
                continue

            edge = (current_vertex, neighbor)
            edge_index = edge_to_index.get(edge)
            if edge_index is not None:
                weight=gradient_F_t[edge_index]
                #weight = edge_weights_matrix[edge_index, iteration]  # Get weight for the current iteration
                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_vertices[neighbor] = current_vertex
                    heapq.heappush(queue, (new_distance, neighbor))

    return distances, previous_vertices

# Function to reconstruct the shortest path
def get_shortest_path(previous_vertices, start, end):
    path = []
    current_vertex = end
    while current_vertex is not None:
        path.append(current_vertex)
        current_vertex = previous_vertices[current_vertex]

    path.reverse()
    if path[0] == start:
        return path
    else:
        return []  # No path found

# Build a connected graph with random edge weights
def build_connected_graph(num_vertices, num_edges):
    graph = {str(i): {} for i in range(num_vertices)}

    # Ensure the graph is connected by adding edges between adjacent vertices
    for i in range(num_vertices - 1):
        weight = random.randint(1, 10)  # Assign a random weight between 1 and 10
        graph[str(i)][str(i + 1)] = weight  # Directed edge i -> i + 1

    # Add random directed edges between non-adjacent vertices
    while sum(len(neighbors) for neighbors in graph.values()) < num_edges:
        v1 = random.choice(list(graph.keys()))
        v2 = random.choice(list(graph.keys()))

        if v1 != v2 and v2 not in graph[v1]:
            weight = random.randint(1, 10)
            graph[v1][v2] = weight  # Directed edge from v1 to v2

    return graph

# Generate a matrix of random edge weights: shape (num_edges x num_iterations)
def generate_random_edge_weights_matrix(graph, num_iterations):
    edge_list, edge_to_index = get_edge_list_and_index(graph)
    num_edges = len(edge_list)

    # Create a matrix where each row is an edge and each column is a random weight for an iteration
    edge_weights_matrix = np.random.randint(1, 11, size=(num_edges, num_iterations))  # Random weights between 1 and 10
    
    return edge_weights_matrix, edge_to_index

# Get the edge list and map edges to an index
def get_edge_list_and_index(graph):
    edge_list = []
    edge_to_index = {}
    index = 0
    for vertex in graph:
        for neighbor, weight in graph[vertex].items():
            edge = (vertex, neighbor)
            if edge not in edge_to_index:
                edge_list.append(edge)
                edge_to_index[edge] = index
                index += 1
    return edge_list, edge_to_index

# Initialize x_1 to a random path from start_vertex to end_vertex
def initialize_random_path(graph, start_vertex, end_vertex, edge_to_index):
    path = []
    current_vertex = start_vertex
    visited = set([start_vertex])

    # Randomly walk through the graph until reaching the end_vertex
    while current_vertex != end_vertex:
        neighbors = list(graph[current_vertex].keys())
        unvisited_neighbors = [v for v in neighbors if v not in visited]
        
        if not unvisited_neighbors:
            print("No path found!")
            return np.zeros(len(edge_to_index))  # Return a zero vector if no path exists

        next_vertex = random.choice(unvisited_neighbors)
        path.append((current_vertex, next_vertex))
        visited.add(next_vertex)
        current_vertex = next_vertex

    # Initialize x_1 as an edge vector based on the random path
    edge_vector = np.zeros(len(edge_to_index), dtype=int)

    # Convert the path to a binary edge vector
    for edge in path:
        edge_index = edge_to_index.get(edge)
        if edge_index is not None:
            edge_vector[edge_index] = 1

    return edge_vector

# Online Conditional Gradient 
def online_conditional_gradient(graph, start_vertex, end_vertex,eta, num_iterations=10):
    # Generate the random edge weights matrix (num_edges x num_iterations)
    edge_weights_matrix, edge_to_index = generate_random_edge_weights_matrix(graph, num_iterations)
    print(f"Generated random edge weights matrix:\n{edge_weights_matrix}")
    distances, previous_vertices = djikstra(graph, start_vertex, np.sum(edge_weights_matrix, axis=1), edge_to_index)
    if distances[end_vertex] == float('inf'):
      print(f"No path exists from {start_vertex} to {end_vertex}.")
    
    path_offline = get_shortest_path(previous_vertices, start_vertex, end_vertex)
    edge_vector_offline_optimum = np.zeros(len(edge_to_index), dtype=int)

    # Convert the path to a binary edge vector
    for edge in path_offline:
      edge_index = edge_to_index.get(edge)
      if edge_index is not None:
        edge_vector_offline_optimum[edge_index] = 1
    # Initialize x_1 (random path as edge vector)
    x_1 = initialize_random_path(graph, start_vertex, end_vertex, edge_to_index)
    print(f"Initial x_1 (random path edge vector): {x_1}")
    x=x_1
    x=np.array(x)
    x_1=np.array(x_1)
    lambd=1/(2*(num_iterations**0.5))
    Q=np.zeros(num_iterations)
    grad_g=np.zeros((edge_weights_matrix.shape[0],num_iterations))
    #gradients_cumulative = np.zeros(edge_weights_matrix.shape[0])  # Cumulative gradient vector (size of number of edges)
    eta_t=eta/(2*(num_iterations**(3/4)))
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")
        grad_F_t=np.zeros(edge_weights_matrix.shape[0])
        for i in range(iteration+1):
          grad_F_t=edge_weights_matrix[:,i]+grad_F_t+(lambd**2)*np.exp(lambd*Q[i])*grad_g[:,i]
          #print(F_t.shape)
        # Regularization term: (norm(x - x1)^2) / eta
        L_t=np.linalg.norm(edge_weights_matrix[:,iteration])
        regularization = 2*(x-x_1) / (eta_t/L_t)
        grad_F_t=grad_F_t+regularization
        #print(f"Regularization term: {regularization}")

        # Run Dijkstra on the graph with the current iteration's edge weights
        distances, previous_vertices = djikstra(graph, start_vertex, grad_F_t, edge_to_index)

        # Check if there is a path to the end vertex
        if distances[end_vertex] == float('inf'):
            print(f"No path exists from {start_vertex} to {end_vertex}.")
            break

        # Get the shortest path
        sigma=min(1,2/((iteration+1)**0.5))
        path = get_shortest_path(previous_vertices, start_vertex, end_vertex)
        edge_vector = np.zeros(len(edge_to_index), dtype=int)

        # Convert the path to a binary edge vector
        for edge in path:
          edge_index = edge_to_index.get(edge)
          if edge_index is not None:
            edge_vector[edge_index] = 1

        #path= np.array(path, dtype=np.float64)
        grad_g[:,iteration]=np.abs(edge_vector - edge_vector_offline_optimum)
        x=(1-sigma)*x+sigma*edge_vector
        if iteration>0:
          Q[iteration]=np.sum(grad_g[:,iteration],axis=0)+Q[iteration-1]

        print(f"Shortest distance from {start_vertex} to {end_vertex}: {distances[end_vertex]}")
        #print(f"Shortest path: {' -> '.join(path)}")

    return graph

# Example inputs for testing
num_vertices = 100
num_edges = 3000
num_iterations = 10

# Ensure that the number of vertices is greater than 1 and there are enough edges
if num_vertices <= 1 or num_edges < num_vertices - 1:
    print("Graph should have more than 1 vertex and enough edges to be connected.")

# Build the graph with the specified number of vertices and edges
graph = build_connected_graph(num_vertices, num_edges)

print("Initial graph:", graph)

# Input start and end vertices (for testing, hardcoded)
start_vertex = '4'
end_vertex = '49'

# Ensure start and end are valid vertices in the graph
if start_vertex not in graph or end_vertex not in graph:
    print("Invalid start or end vertex.")

# Run the online conditional gradient 
graph = online_conditional_gradient(graph, start_vertex, end_vertex, num_edges,num_iterations=num_iterations)

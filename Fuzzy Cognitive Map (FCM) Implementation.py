import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time


MIN_N = 3
MAX_N = 10


def adj_matrix(n: int) -> list[list[int]]:
    """
    Creates an n x n adjacency matrix.
    """
    adj = np.random.uniform(-1, 1, (n, n))

    for i in range(n):
        adj[i][i] = 0 # diagonal is zeros
    print(adj)
    return adj


def sigmoid(x: int) -> int:
    """
    Sigmoid function:
      sigmoid(x) = 1/(1 + exp(-x))
    """
    return 1/(1 + np.exp(-x))


def update_nodes(n: int, adj_Matrix: list[list[int]], values: list[int]) -> list[int]:
    """
    Calculate node value.
       V_i(t + 1) = f(V_i(t) + summation of (W_ji * V_j))
       f is the sigmoid function in this case
    """
    flag = np.zeros(n)
    MAX_ITER = 100
    epsilon = 10**-5
    
    for i in range(MAX_ITER):
        for j in range(len(values)):
            if not flag[j]:
                total = 0
                old = values[j]
                for k in range(n):
                    total += (values[k] * adj_Matrix[k][j])

                values[j] = sigmoid(old + total)
                
                if (abs(values[j] - old) < epsilon):
                    flag[j] = 1
        
        # if all have converged
        if np.all(flag == 1):
            break

    return values


def show_graph(adj):
    """
    Create the graph visualization
    """
    g = nx.from_numpy_array(adj, create_using=nx.MultiDiGraph)

    edges = {(i,j): round(adj[i][j], 2) for i in range(len(adj)) for j in range(len(adj[0])) if adj[i][j]}
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edges)
    plt.show()


def edit_node(values):
    """
    Allow the user to edit their node value.
    """
    print("Please select which of the following nodes to change:")
    print("\t\n".join(map(str, np.round(values, 4))))

    for i in range(len(values)):
        print(f"Node: {i + 1}; Value: {np.round(values[i], 4)}\n")

    selection = input("Node to change: ")

    choice = input(f"New value of node {selection}: ")

    values[int(selection) - 1] = float(choice)


def edit_edge(adj):
    """
    Edit an edge between two nodes.
    """
    print("Please view the graph and select edges that you wish to change. Showing Visualization...")
    time.sleep(2)
    show_graph(adj)
    source = int(input("Input the source node number: "))
    destination = int(input("Input the destination node number: "))
    new = float(input("Enter a new value for the edge: "))

    adj[source][destination] = new

    show_graph(adj)


def main():
    """
    Main function, controls the CLI.
    """
    print("Starting program...")
    n = np.random.randint(MIN_N, MAX_N)

    print("Creating the adjacency matrix...")
    adj = adj_matrix(n)
    print(f"Successfully created the {n} x {n} adjacency matrix.")

    print("Calculating the values of each node.")
    values = update_nodes(n, adj, np.random.rand(n))

    while True:
        selection = input("Menu:\n\t1. Change a node's value\n\t2. Change an edge value\n\t3. Display a visual representation of the graph\n\t4. Quit\nSelection: ")
        if (selection.isnumeric()):
            selection = int(selection)
            match selection:
                case 1: edit_node(values)
                case 2: edit_edge(adj)
                case 3: show_graph(adj)
                case 4: break
        else:
            print("Please select a valid action.\n")
    

if __name__ == "__main__":
    main()


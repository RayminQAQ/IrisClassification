import matplotlib.pyplot as plt
import os

# Function to plot a line graph and save it as a PNG file
def plot_epoch_data(epoch_data, input_data, save_path=None, filename="plot.png", title="Epoch vs Metric", xlabel="Epoch", ylabel="Value"):
    """
    This function plots a line graph for given epoch data and corresponding metric data,
    and saves it as a PNG file if a save path is provided.
    
    Parameters:
    - epoch_data (list or array): Data for the X-axis (Epochs).
    - input_data (list or array): Data for the Y-axis (Metric values).
    - save_path (str): Directory to save the PNG file (if None, the plot will not be saved).
    - filename (str): Name of the file to save the plot as PNG.
    - title (str): Title of the graph.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    """
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(epoch_data, input_data, marker='o', linestyle='-', color='b')  # Plot the data with markers
    plt.title(title)  # Set the title of the graph
    plt.xlabel(xlabel)  # Set the label for the X-axis
    plt.ylabel(ylabel)  # Set the label for the Y-axis
    plt.grid(True)  # Enable the grid for better readability
    plt.xticks(epoch_data)  # Set ticks on X-axis to show all epoch values

    # Check if the save_path directory exists
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the directory if it does not exist
            print(f"Directory created: {save_path}")

        # Save the plot
        plt.savefig(f"{save_path}/{filename}", format='png', bbox_inches='tight')
        print(f"Plot saved as {save_path}/{filename}")
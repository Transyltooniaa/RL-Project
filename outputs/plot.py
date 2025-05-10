import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from pathlib import Path


class DataExplorer:
    def __init__(self, base_dir='outputs'):
        """Initialize with the base directory containing scenario data."""
        self.base_dir = Path(base_dir)
        self.df = None
        self.current_scenario = None
        self.current_episode = None
        
    def get_scenarios(self):
        """Return a list of available scenario directories."""
        try:
            return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        except FileNotFoundError:
            print(f"Error: Base directory '{self.base_dir}' not found.")
            return []
            
    def get_episodes(self, scenario):
        """Return a list of CSV files in the given scenario directory."""
        scenario_path = self.base_dir / scenario
        try:
            return [f for f in os.listdir(scenario_path) if f.endswith('.csv')]
        except FileNotFoundError:
            print(f"Error: Scenario directory '{scenario_path}' not found.")
            return []
    
    def load_data(self, scenario, episode):
        """Load data from the specified episode file."""
        file_path = self.base_dir / scenario / episode
        try:
            self.df = pd.read_csv(file_path)
            self.current_scenario = scenario
            self.current_episode = episode
            print(f"Loaded data: {len(self.df)} rows × {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"Error loading data from '{file_path}': {e}")
            return False
    
    def get_numeric_columns(self):
        """Return a list of numeric columns in the loaded dataframe."""
        if self.df is None:
            return []
        return [col for col, dtype in zip(self.df.columns, self.df.dtypes) 
                if np.issubdtype(dtype, np.number)]
    
    def get_all_columns(self):
        """Return all columns in the loaded dataframe."""
        if self.df is None:
            return []
        return list(self.df.columns)
    
    def generate_plot(self, plot_type, x_col, y_cols, title=None, **kwargs):
        """Generate the specified type of plot."""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return False
        
        if x_col not in self.df.columns:
            print(f"Column '{x_col}' not found in the dataframe.")
            return False
            
        # Validate y columns
        if isinstance(y_cols, list):
            for col in y_cols:
                if col not in self.df.columns:
                    print(f"Column '{col}' not found in the dataframe.")
                    return False
        elif y_cols not in self.df.columns:
            print(f"Column '{y_cols}' not found in the dataframe.")
            return False
            
        # Create figure with good size and resolution
        plt.figure(figsize=(10, 6), dpi=100)
        
        # Generate the specified plot type
        if plot_type == 'line':
            self._create_line_plot(x_col, y_cols, **kwargs)
        elif plot_type == 'scatter':
            self._create_scatter_plot(x_col, y_cols, **kwargs)
        elif plot_type == 'bar':
            self._create_bar_plot(x_col, y_cols, **kwargs)
        elif plot_type == 'histogram':
            self._create_histogram(x_col, **kwargs)
        elif plot_type == 'heatmap':
            self._create_heatmap(x_col, y_cols, **kwargs)
        else:
            print(f"Unsupported plot type: {plot_type}")
            plt.close()
            return False
            
        # Set title
        if title:
            plt.title(title, fontsize=14)
        else:
            y_label = ", ".join(y_cols) if isinstance(y_cols, list) else y_cols
            plt.title(f"{y_label} vs {x_col} ({self.current_scenario} - {self.current_episode})", fontsize=14)
            
        # Final plot configuration
        plt.xlabel(x_col, fontsize=12)
        if plot_type != 'heatmap' and plot_type != 'histogram':
            plt.ylabel(y_cols if not isinstance(y_cols, list) else "Value", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return True
    
    def _create_line_plot(self, x_col, y_cols, **kwargs):
        """Create a line plot."""
        if isinstance(y_cols, list):
            for col in y_cols:
                plt.plot(self.df[x_col], self.df[col], marker=kwargs.get('marker', '.'), 
                         linestyle=kwargs.get('linestyle', '-'), label=col)
            plt.legend()
        else:
            plt.plot(self.df[x_col], self.df[y_cols], marker=kwargs.get('marker', '.'), 
                     linestyle=kwargs.get('linestyle', '-'))
    
    def _create_scatter_plot(self, x_col, y_cols, **kwargs):
        """Create a scatter plot."""
        if isinstance(y_cols, list):
            for col in y_cols:
                plt.scatter(self.df[x_col], self.df[col], alpha=kwargs.get('alpha', 0.7), 
                           s=kwargs.get('size', 30), label=col)
            plt.legend()
        else:
            color_col = kwargs.get('color_by')
            if color_col and color_col in self.df.columns:
                # Create a colorful scatter plot based on a third variable
                scatter = plt.scatter(self.df[x_col], self.df[y_cols], 
                                     c=self.df[color_col], alpha=kwargs.get('alpha', 0.7),
                                     s=kwargs.get('size', 30), cmap='viridis')
                plt.colorbar(scatter, label=color_col)
            else:
                plt.scatter(self.df[x_col], self.df[y_cols], alpha=kwargs.get('alpha', 0.7),
                           s=kwargs.get('size', 30))
    
    def _create_bar_plot(self, x_col, y_cols, **kwargs):
        """Create a bar plot."""
        if isinstance(y_cols, list):
            x_positions = np.arange(len(self.df))
            width = 0.8 / len(y_cols)
            
            for i, col in enumerate(y_cols):
                offset = (i - len(y_cols)/2 + 0.5) * width
                plt.bar(x_positions + offset, self.df[col], width=width, label=col)
                
            plt.xticks(x_positions, self.df[x_col], rotation=kwargs.get('rotation', 45))
            plt.legend()
        else:
            plt.bar(self.df[x_col], self.df[y_cols], alpha=kwargs.get('alpha', 0.7))
            plt.xticks(rotation=kwargs.get('rotation', 45))
    
    def _create_histogram(self, x_col, **kwargs):
        """Create a histogram."""
        bins = kwargs.get('bins', 'auto')
        plt.hist(self.df[x_col], bins=bins, alpha=0.7, color=kwargs.get('color', 'skyblue'),
                edgecolor=kwargs.get('edgecolor', 'black'))
        plt.ylabel('Frequency', fontsize=12)
    
    def _create_heatmap(self, x_col, y_col, **kwargs):
        """Create a heatmap."""
        if not isinstance(y_col, str):
            print("Heatmap requires a single y column.")
            return
            
        z_col = kwargs.get('z_col')
        if not z_col or z_col not in self.df.columns:
            print("Heatmap requires a valid z column to determine color.")
            return
            
        # Create pivot table for heatmap
        try:
            pivot_table = self.df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='mean')
            
            # Create a nice colormap
            colors = [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0)]  # Blue -> Green -> Red
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
            
            im = plt.imshow(pivot_table, cmap=cmap, aspect='auto')
            plt.colorbar(im, label=z_col)
            
            # Set x and y ticks
            plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=90)
            plt.yticks(range(len(pivot_table.index)), pivot_table.index)
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            print("Heatmap requires categorical or discrete data for x and y columns.")
    
    def save_plot(self, filename=None):
        """Save the current plot to a file."""
        if filename is None:
            filename = f"{self.current_scenario}_{self.current_episode.replace('.csv', '')}_plot.png"
        
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving plot: {e}")
            return False
    
    def show_plot(self):
        """Display the current plot."""
        plt.show()
    
    def get_summary_stats(self, columns=None):
        """Return summary statistics for the specified columns."""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return None
            
        if columns is None:
            columns = self.get_numeric_columns()
        elif isinstance(columns, str):
            columns = [columns]
            
        try:
            stats = self.df[columns].describe().T
            return stats
        except Exception as e:
            print(f"Error generating summary statistics: {e}")
            return None


def choose_option(options, prompt):
    """Prompt the user to choose one of the options by number."""
    if not options:
        print("No options available.")
        return None
        
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        choice = input(prompt)
        if choice.lower() == 'q':
            return 'quit'
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(options):
            print(f"Please enter a number between 1 and {len(options)} or 'q' to quit.")
        else:
            return options[int(choice) - 1]


def choose_multiple(options, prompt):
    """Allow the user to select multiple options."""
    if not options:
        print("No options available.")
        return []
        
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    print("  Enter numbers separated by spaces (e.g., '1 3 5') or 'all' for all options.")
    while True:
        choice = input(prompt)
        if choice.lower() == 'q':
            return 'quit'
        if choice.lower() == 'all':
            return options
        
        try:
            indices = [int(x.strip()) for x in choice.split()]
            if all(1 <= idx <= len(options) for idx in indices):
                return [options[idx-1] for idx in indices]
            else:
                print(f"Please enter valid numbers between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")


def interactive_mode(explorer):
    """Run the application in interactive mode."""
    scenarios = explorer.get_scenarios()
    if not scenarios:
        print("No scenarios found. Exiting.")
        return
    
    print("\n===== DATA EXPLORER =====")
    print("Available scenarios:")
    scenario = choose_option(scenarios, "Select scenario by number (or 'q' to quit): ")
    if scenario == 'quit':
        return
    
    episodes = explorer.get_episodes(scenario)
    if not episodes:
        print("No episodes found in the selected scenario.")
        return
    
    print(f"\nAvailable episodes in '{scenario}':")
    episode = choose_option(episodes, "Select episode by number (or 'q' to quit): ")
    if episode == 'quit':
        return
    
    if not explorer.load_data(scenario, episode):
        return
    
    while True:
        print("\n===== PLOT OPTIONS =====")
        print("  1. Line plot")
        print("  2. Scatter plot")
        print("  3. Bar chart")
        print("  4. Histogram")
        print("  5. Heatmap")
        print("  6. Show data summary")
        print("  7. Load different data")
        print("  8. Exit")
        
        choice = input("Select an option by number: ")
        if not choice.isdigit() or int(choice) < 1 or int(choice) > 8:
            print("Invalid choice. Please enter a number between 1 and 8.")
            continue
        
        choice = int(choice)
        
        if choice == 8:  # Exit
            break
        
        if choice == 7:  # Load different data
            return interactive_mode(explorer)
        
        if choice == 6:  # Show data summary
            columns = explorer.get_numeric_columns()
            print("\nAvailable numeric columns:")
            selected_cols = choose_multiple(columns, "Select columns for summary stats (or 'q' to cancel): ")
            if selected_cols == 'quit':
                continue
                
            stats = explorer.get_summary_stats(selected_cols)
            if stats is not None:
                print("\n===== SUMMARY STATISTICS =====")
                pd.set_option('display.precision', 3)
                print(stats)
            continue
        
        # For plotting options
        plot_types = ['line', 'scatter', 'bar', 'histogram', 'heatmap']
        plot_type = plot_types[choice - 1]
        
        # Get columns for plotting
        all_columns = explorer.get_all_columns()
        numeric_columns = explorer.get_numeric_columns()
        
        print("\nAvailable columns:")
        x_col = choose_option(all_columns if plot_type in ['bar', 'heatmap'] else numeric_columns, 
                             "Select column for x-axis (or 'q' to cancel): ")
        if x_col == 'quit':
            continue
        
        if plot_type == 'histogram':
            # Histogram only needs one column
            y_cols = None
            kwargs = {'bins': input("Number of bins (default is 'auto'): ") or 'auto'}
        elif plot_type == 'heatmap':
            # Heatmap needs x, y, and z columns
            print("\nSelect column for y-axis:")
            y_col = choose_option(all_columns, "Select column for y-axis (or 'q' to cancel): ")
            if y_col == 'quit':
                continue
                
            print("\nSelect column for color values (z-axis):")
            z_col = choose_option(numeric_columns, "Select column for color values (or 'q' to cancel): ")
            if z_col == 'quit':
                continue
                
            y_cols = y_col
            kwargs = {'z_col': z_col}
        else:
            # Other plots can have multiple y columns
            print("\nAvailable columns for y-axis:")
            y_cols = choose_multiple(numeric_columns, "Select column(s) for y-axis (or 'q' to cancel): ")
            if y_cols == 'quit':
                continue
                
            # Additional options based on plot type
            kwargs = {}
            if plot_type == 'scatter':
                print("\nOptional: Select a column for color coding:")
                print("  0. None (use default coloring)")
                color_options = ["None"] + numeric_columns
                color_col = choose_option(color_options, "Select column for color (or 'q' to cancel): ")
                if color_col == 'quit':
                    continue
                if color_col != "None":
                    kwargs['color_by'] = color_col
        
        # Generate the plot
        if explorer.generate_plot(plot_type, x_col, y_cols, **kwargs):
            # Ask if the user wants to save the plot
            save_choice = input("Do you want to save this plot? (y/n): ").lower()
            if save_choice == 'y':
                filename = input("Enter filename (leave blank for default): ")
                explorer.save_plot(filename if filename else None)
            
            # Show the plot
            explorer.show_plot()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Interactive data explorer for CSV files.')
    parser.add_argument('--base-dir', default='outputs', help='Base directory containing scenario data.')
    parser.add_argument('--scenario', help='Directly specify scenario folder.')
    parser.add_argument('--episode', help='Directly specify episode file.')
    parser.add_argument('--batch', action='store_true', help='Run in batch mode using provided options.')
    parser.add_argument('--x-col', help='Column for x-axis in batch mode.')
    parser.add_argument('--y-cols', help='Comma-separated list of columns for y-axis in batch mode.')
    parser.add_argument('--plot-type', default='line', choices=['line', 'scatter', 'bar', 'histogram', 'heatmap'],
                       help='Type of plot to generate in batch mode.')
    parser.add_argument('--output', help='Output file for saving the plot in batch mode.')
    return parser.parse_args()


def batch_mode(explorer, args):
    """Run the application in batch mode using command-line arguments."""
    # Load the data
    if not explorer.load_data(args.scenario, args.episode):
        return
    
    # Parse y columns
    y_cols = args.y_cols.split(',') if ',' in args.y_cols else args.y_cols
    
    # Generate the plot
    kwargs = {}
    if args.plot_type == 'heatmap' and isinstance(y_cols, str):
        # For heatmap, we need a z column
        numeric_cols = explorer.get_numeric_columns()
        if len(numeric_cols) > 0 and numeric_cols[0] != args.x_col and numeric_cols[0] != y_cols:
            kwargs['z_col'] = numeric_cols[0]
        else:
            print("Error: Heatmap requires a third numeric column for z values.")
            return
    
    if explorer.generate_plot(args.plot_type, args.x_col, y_cols, **kwargs):
        # Save the plot
        explorer.save_plot(args.output)
        
        # Show the plot if no output file is specified
        if not args.output:
            explorer.show_plot()


def main():
    """Main function to run the data explorer."""
    args = parse_args()
    explorer = DataExplorer(base_dir=args.base_dir)
    
    if args.batch and args.scenario and args.episode and args.x_col and args.y_cols:
        # Run in batch mode
        batch_mode(explorer, args)
    else:
        # Run in interactive mode
        interactive_mode(explorer)


if __name__ == '__main__':
    main()
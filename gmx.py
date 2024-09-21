import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import display, HTML

def gmx(executable, arguments, inputs, outputs, name = None, path = None, ensemble = None, scheduler = None, wait = False, stdin = None):
    """
    Run a GROMACS command using the specified executable, input files, arguments, and output files.

    Args:
        executable  (str): The GROMACS executable to run.
        input_files (dict): A dictionary mapping input file flags to their corresponding file paths.
        arguments (list): A list of additional arguments to pass to the GROMACS command.
        output_files (dict): A dictionary mapping output file flags to their corresponding file paths.
        working_dir (str, optional): The working directory to run the command in. Defaults to None.
        toString (bool, optional): If True, return the command as a string instead of running it. Defaults to False.

    Returns:
        int or str: The exit code of the GROMACS command, or the command as a string if toString is True.
    """
    
    # Start with the main command
    cmd = ["gmx", executable]

    if executable == 'mdrun':
        cmd = ['mpirun', "gmx_mpi", executable]
    
    # Add arguments to the command
    cmd.extend(arguments)
    
    # Add input files to the command
    for flag, file in inputs.items():
        cmd.extend([flag, file])

    # Add output files to the command
    for flag, file in outputs.items():
        cmd.extend([flag, file])

    # No molecular dynamics simulation
    if executable != 'mdrun':
        # Run the command
        process = subprocess.run(cmd, capture_output=True, text=True, cwd=path, input=stdin)
    
        # Print the output
        print(process.stdout)
        print(process.stderr)

    else:
        if ensemble is None:
            raise ValueError("ensemble must be provided.")
        
        if name is None:
            raise ValueError("name must be provided.")
        
        if path is None:
            raise ValueError("path must be provided.")
        
    
        schdlr = {"job_name":"%s%s" % (name, ensemble),  
                  "output":"%s.slurm.log" % ensemble, 
                  "partition":"andromeda", 
                  "nodes" :"1-12", 
                  "ntasks_per_node":"1",    
                  "ntasks":"10",
                  "modules":["module load mpi/mpich-x86_64",
                             "module load gromacs/2023.5-plumed"]}

        if scheduler is not None:
            for key in scheduler:
                schdlr[key] = scheduler[key]

        batch_script = [
            "#!/bin/bash",
            "",
            "#SBATCH --job-name=%s" % schdlr['job_name'],
            "#SBATCH --output=%s" % schdlr['output'],
            ("#SBATCH --partition=%s" % schdlr['partition']) if schdlr['partition'] is not None else '',
            ("#SBATCH --nodes=%s" % schdlr['nodes']) if schdlr['nodes'] is not None else '',
            "#SBATCH --ntasks=%s" % schdlr['ntasks'],
            ("#SBATCH --ntasks-per-node=%s" % schdlr['ntasks_per_node']) if schdlr['ntasks_per_node'] is not None else '',
            "",
            "# Required modules"] + schdlr['modules'] + [
            "",
            "# Run the molecular dynamics simulation",
            ' '.join(cmd)
        ]

        # Save the batch script to a file
        with open("%s/%s.slurm" % (path, ensemble), 'w') as file:
            file.write("\n".join(batch_script))

        # Submit the batch script to the cluster
        sbmttr_cmd = ["sbatch",'%s.slurm' % ensemble]
        if wait:
            sbmttr_cmd = ["sbatch","--wait",'%s.slurm' % ensemble]
        
        subprocess.run(sbmttr_cmd, cwd = path)


def xvg_line (title, subtitle, path, ensemble, xlabel, ylabel, label, sufix = '', movavg = 0):
    """
    Generates and saves a plot from data in an .xvg file, with optional moving average.

    This function reads data from a specified .xvg file, generates a plot of the data,
    and optionally overlays a moving average. The plot is saved as a PNG file.

    Parameters:
    - title (str): The main title of the plot.
    - subtitle (str): The subtitle of the plot.
    - path (str): The directory path where the .xvg file is located and where the PNG file will be saved.
    - ensemble (str): The name of the ensemble or dataset to be plotted, used to identify the .xvg file.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis. If it contains '$', the y-axis labels will be formatted as scientific notation.
    - label (str): The legend label for the plotted data.
    - sufix (str, optional): A suffix to append to the ensemble name for identifying the .xvg file. Defaults to ''.
    - movavg (int, optional): The window size for the moving average. If 0, no moving average is plotted. Defaults to 0.

    Returns:
    None. The function saves the plot as a PNG file in the specified path and does not return any value.

    Example:
    xvg_('Temperature Profile', 'Simulation over Time', '/data/simulations', 'temp', 'Time (ps)', 'Temperature (K)', 'Run 1', sufix='_run1', movavg=10)
    This will read the data from '/data/simulations/temp_run1.xvg', generate a plot with a moving average of window size 10,
    and save it as '/data/simulations/temp_run1.png'.
    """

    # Read the data from the .xvg file
    data = np.loadtxt('%s/%s%s.xvg' % (path, ensemble, sufix), comments=['@', '#'])

    # Plot the data
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.plot(data[:, 0], data[:, 1], c = "black", lw = 2, label = label) 

    # Calculate the moving average
    if movavg != 0:
        window_size = movavg  # Set the window size to 10
        window = np.ones(window_size) / window_size  # Create a uniform window
        moving_avg = np.convolve(data[:, 1], window, 'valid')  # Calculate the moving average
        ax.plot(data[window_size - 1:, 0], moving_avg, c = "red", lw = 2, label = '10 ps Moving average')  # Add the moving average line

    plt.suptitle(title)
    plt.title(subtitle)
    plt.xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize = 10)

    # Set the y-axis label formatter
    if("$" in ylabel):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: r'${:.2e}$'.format(value)))

    # Add a legend to the plot
    ax.legend(loc='upper right')

    # Save the plot as a PNG file
    plt.savefig('%s/%s%s.png' % (path, ensemble, sufix), dpi = 300)

    # Close the figure
    plt.close(fig)

def xvg_multi_line(title, path, ensemble, xlabel, ylabel, label_prefix, replicas, sufix='', movavg=0, plot_type='values'):
    """
    Generates and saves a single plot with data from multiple replicas, each in a different color.

    Parameters:
    - title (str): The main title of the plot.
    - path (str): The directory path where the .xvg files are located.
    - ensemble (str): The name of the ensemble or dataset to be plotted.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - label_prefix (str): The prefix for the legend label for the plotted data.
    - replicas (int): The number of replicas to generate plots for.
    - sufix (str, optional): A suffix to append to the ensemble name for identifying the .xvg file. Defaults to ''.
    - movavg (int, optional): The window size for the moving average. If 0, no moving average is plotted. Defaults to 0.
    - plot_type (str, optional): Controls what to plot: 'values', 'moving_average', or 'both'. Defaults to 'values'.

    Returns:
    None. The function saves the plot as a PNG file in the specified path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, replicas))

    for i in range(replicas):
        data_path = '%s/replica%02d%s.xvg' % (path, i + 1, sufix)
        data = np.loadtxt(data_path, comments=['@', '#'])
        label = '%s %d' % (label_prefix, i + 1)

        if plot_type in ['both', 'values']:
            ax.plot(data[:, 0], data[:, 1], c=colors[i], lw=2, label=label)

        if movavg != 0 and plot_type in ['both', 'moving_average']:
            window = np.ones(movavg) / movavg
            moving_avg = np.convolve(data[:, 1], window, 'valid')
            ax.plot(data[movavg - 1:, 0], moving_avg, '--', c=colors[i], lw=2, label='%s MA' % label)

    plt.suptitle(title)
    plt.xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=10)

    if("$" in ylabel):
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: r'${:.2e}$'.format(value)))

    ax.legend(loc='upper right')
    plt.savefig('%s/%s%s.png' % (path, ensemble, sufix), dpi = 300)
    plt.close(fig)

def xvg_multi_density(title, path, ensemble, xlabel, ylabel, label_prefix, replicas, sufix='', bandwidth=None):
    """
    Generates and saves a single plot with density plots from multiple replicas, each in a different color.

    Parameters:
    - title (str): The main title of the plot.
    - path (str): The directory path where the .xvg files are located.
    - ensemble (str): The name of the ensemble or dataset to be plotted.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - label_prefix (str): The prefix for the legend label for the plotted data.
    - replicas (int): The number of replicas to generate plots for.
    - sufix (str, optional): A suffix to append to the ensemble name for identifying the .xvg file. Defaults to ''.
    - bandwidth (float, optional): The bandwidth for the kernel density estimate. Smaller values produce more detailed plots. Defaults to None, which automatically selects the bandwidth.

    Returns:
    None. The function saves the plot as a PNG file in the specified path.
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, replicas))

    for i in range(replicas):
        data_path = f'{path}/replica{i + 1:02d}{sufix}.xvg'
        data = np.loadtxt(data_path, comments=['@', '#'])
        sns.kdeplot(data[:, 1], color=colors[i], bw_adjust=bandwidth, label=f'{label_prefix} {i + 1}')

    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend(loc='upper right')
    plt.savefig(f'{path}/{ensemble}{sufix}_density.png', dpi=300)
    plt.close()

def xvg(title, path, ensemble, xlabel, ylabel, label, measure = None, units = None, values = None, replicas = None, sufix = '', movavg = 0, multi_lines = False, plot_type = 'values'):
    """
    Generates and displays plots from .xvg files for a given ensemble, optionally across multiple replicas.

    This function creates plots for data contained in .xvg files. It supports generating plots for single
    simulations or multiple replicas. For multiple replicas, it can display additional information such as
    specific measures and units. The plots are displayed in an HTML table format. Each plot can optionally
    include a moving average of the data.

    Parameters:
    - title (str): The main title of the plot.
    - path (str): The directory path where the .xvg files are located.
    - ensemble (str): The name of the ensemble or dataset to be plotted.
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - label (str): The legend label for the plotted data.
    - measure (str, optional): The physical measure represented in the plot, relevant for replicas. Defaults to None.
    - units (str, optional): The units of the measure, relevant for replicas. Defaults to None.
    - values (list, optional): A list of values corresponding to the measure for each replica. Defaults to None.
    - replicas (int, optional): The number of replicas to generate plots for. If None, a single plot is generated. Defaults to None.
    - sufix (str, optional): A suffix to append to the ensemble name for identifying the .xvg file. Defaults to ''.
    - movavg (int, optional): The window size for the moving average. If 0, no moving average is plotted. Defaults to 0.

    Returns:
    None. The function generates and displays the plots in an HTML table format within the Jupyter Notebook environment.

    Example:
    xvg('Temperature Profile', '/data/simulations', 'temp', 'Time (ps)', 'Temperature (K)', 'Run 1', measure='Temperature', units='K', values=[300, 310], replicas=2, sufix='_run1', movavg=10)
    This will generate and display plots for two replicas of a temperature profile, each with a moving average of window size 10.
    """
     
    # Create the HTML table for displaying the plots
    html = '<table>'

    if replicas is None:
        subtitle = ""
        xvg_line(title, subtitle, path, ensemble, xlabel, ylabel, label, sufix, movavg)
        html += '<td><img src="{}?{}" style="width:100%"></td>'.format('%s/%s%s.png' % (path, ensemble,sufix), time.time())
    else:
        if multi_lines:
            xvg_multi_line(title, path, ensemble, xlabel, ylabel, label, replicas, sufix, movavg, plot_type)
            html += '<td><img src="{}?{}" style="width:100%"></td>'.format('%s/%s%s.png' % (path, ensemble,sufix), time.time())
        else:
            for i in range(1, replicas):
                npath = "%s/replica%02d" % (path, i)

                if ensemble == 'min':
                    subtitle = 'Replica #%02d' % (i - 1)
                else:
                    subtitle = 'Replica #%02d (%s = %0.2f%s)' % (i - 1, measure, values[i - 1], units)

                xvg_line(title, subtitle, npath, ensemble, xlabel, ylabel, label, sufix, movavg)
                html += '<td><img src="{}?{}" style="width:100%"></td>'.format('%s/%s%s.png' % (npath, ensemble,sufix), time.time())
                if i % 3 == 0:
                    html += '</tr>' 

    # Add a query parameter with the current time to the image URL
    html += '</table>'
    display(HTML(html))

import mediapy as media
import numpy as np
import matplotlib.pyplot as plt


import mediapy as media
import numpy as np


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number
    for which the scene cut was detected (i.e. the last frame
    index of the first scene and the first frame index of the second scene)
    """

    # Read the video into frames of (height, width, channels) 3D arrays
    video_frames = media.read_video(video_path)

    # Calculate the cumulative histograms for scene cut detection
    cumulative_histogram = []

    # Standard weights array for the conversion
    grayscale_weights = np.array([0.299, 0.587, 0.114])

    for frame in video_frames:
        # Change it into floats for the dot product
        frame_in_float = frame.astype(np.float32) / 255.0  # Normalize the frame

        # We use the weights to transform the frame into grayscale and sum them up
        # [...,:3] to correctly handle the color channels
        grayscale_frame = np.dot(frame_in_float[..., :3], grayscale_weights)

        # Calculate the cumulative histogram for the grayscale frame
        hist, bins = np.histogram(grayscale_frame, bins=256, range=(0, 1))
        cum_hist = np.cumsum(hist)

        # Store the cumulative histogram to our list
        cumulative_histogram.append(cum_hist)

    # If there is only one scene, return 0, 0
    if len(cumulative_histogram) < 2:
        return 0, 0

    """
    As discussed in the lecture, the scene cut is where the distance 
    is the largest.
    """
    distances = []
    for i in range(len(cumulative_histogram) - 1):
        hist1 = cumulative_histogram[i]
        hist2 = cumulative_histogram[i + 1]

        dist = np.sum(np.abs(hist1 - hist2))

        distances.append(dist)

    cut_index = np.argmax(distances)

    # This is the last frame of scene 1, and the first frame of scene 2
    return int(cut_index), int(cut_index + 1)

def plot_scene_cut_histograms(video_path, cut_index):
    """
    Plot the regular histograms for the frames at the scene cut
    :param video_path: path to the video file
    :param cut_index: index where the scene cut occurs
    """
    # Read the video
    video_frames = media.read_video(video_path)

    # Standard weights array for the conversion
    grayscale_weights = np.array([0.299, 0.587, 0.114])

    # Get the two frames at the cut
    frame1 = video_frames[cut_index]
    frame2 = video_frames[cut_index + 1]

    # Convert to grayscale and get histograms
    gray1 = np.dot(frame1.astype(np.float32)[..., :3] / 255.0, grayscale_weights)
    gray2 = np.dot(frame2.astype(np.float32)[..., :3] / 255.0, grayscale_weights)

    hist1, _ = np.histogram(gray1, bins=256, range=(0, 1))
    hist2, _ = np.histogram(gray2, bins=256, range=(0, 1))

    # Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Add a main title showing which video this is
    fig.suptitle(f'Scene Cut Histograms for {video_path}', fontsize=14, fontweight='bold')

    # Plot histogram for last frame of first scene
    axes[0].bar(range(256), hist1, width=1, edgecolor='none')
    axes[0].set_title(f'Frame {cut_index} (Last frame of Scene 1)')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Pixel Count')
    axes[0].set_xlim([0, 255])

    # Plot histogram for first frame of second scene
    axes[1].bar(range(256), hist2, width=1, edgecolor='none')
    axes[1].set_title(f'Frame {cut_index + 1} (First frame of Scene 2)')
    axes[1].set_xlabel('Intensity')
    axes[1].set_ylabel('Pixel Count')
    axes[1].set_xlim([0, 255])

    plt.tight_layout()
    plt.show()


def generate_distance_plot(video_path, video_name):
    print(f"Generating distance plot for {video_name}...")

    # --- This is the same logic from your main() function ---
    video_frames = media.read_video(video_path)
    cumulative_histogram = []
    grayscale_weights = np.array([0.299, 0.587, 0.114])

    for frame in video_frames:
        frame_in_float = frame.astype(np.float32) / 255.0
        grayscale_frame = np.dot(frame_in_float[..., :3], grayscale_weights)
        hist, bins = np.histogram(grayscale_frame, bins=256, range=(0, 1))
        cum_hist = np.cumsum(hist)
        cumulative_histogram.append(cum_hist)

    distances = []
    for i in range(len(cumulative_histogram) - 1):
        hist1 = cumulative_histogram[i]
        hist2 = cumulative_histogram[i + 1]
        dist = np.sum(np.abs(hist1 - hist2))
        distances.append(dist)

    cut_index = np.argmax(distances)

    # --- This is the new Matplotlib plotting part ---
    plt.figure(figsize=(12, 5))
    plt.plot(distances)
    plt.title(f'Frame-to-Frame Distance for {video_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Cumulative Histogram L1 Distance', fontsize=12)

    # Add a red line to mark the detected cut
    plt.axvline(x=cut_index, color='r', linestyle='--',
                label=f'Detected Cut at Frame {cut_index}')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plot_filename = f'{video_name}_distance_plot.png'
    plt.savefig(plot_filename)
    plt.close()  # Close the figure to save memory
    print(f"Saved plot: {plot_filename}")

def generate_comparison_plot(video_path, video_name):
    print(f"Generating comparison plot for {video_name}...")

    video_frames = media.read_video(video_path)
    grayscale_weights = np.array([0.299, 0.587, 0.114])

    # We will calculate distances for BOTH methods
    cdf_distances = []  # Cumulative (your working method)
    pdf_distances = []  # Non-Cumulative (the "fail" method)

    cumulative_histograms = []
    simple_histograms = []

    for frame in video_frames:
        frame_in_float = frame.astype(np.float32) / 255.0
        grayscale_frame = np.dot(frame_in_float[..., :3], grayscale_weights)
        hist, bins = np.histogram(grayscale_frame, bins=256, range=(0, 1))

        # Method 1: Cumulative Histogram (CDF)
        cum_hist = np.cumsum(hist)
        cumulative_histograms.append(cum_hist)

        # Method 2: Simple Normalized Histogram (PDF)
        simple_hist_pdf = hist / np.sum(hist)
        simple_histograms.append(simple_hist_pdf)

    # Calculate distances for both methods
    for i in range(len(cumulative_histograms) - 1):
        # CDF distance
        cdf_dist = np.sum(np.abs(cumulative_histograms[i] - cumulative_histograms[i + 1]))
        cdf_distances.append(cdf_dist)

        # PDF distance
        pdf_dist = np.sum(np.abs(simple_histograms[i] - simple_histograms[i + 1]))
        pdf_distances.append(pdf_dist)

    # --- Plot them side-by-side ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # 2 rows, 1 column
    fig.suptitle(f'Comparison of Methods for {video_name}', fontsize=16, fontweight='bold')

    # Plot 1: Your working method (CDF)
    ax1.plot(cdf_distances)
    ax1.set_title("Working Method: Cumulative Histogram Distance (CDF)")
    ax1.set_ylabel("L1 Distance")
    ax1.axvline(x=np.argmax(cdf_distances), color='g', linestyle='--', label='Correct Cut')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: The "fail" method (PDF)
    ax2.plot(pdf_distances)
    ax2.set_title("Noisy Method: Simple Histogram Distance (PDF)")
    ax2.set_ylabel("L1 Distance")
    ax2.set_xlabel("Frame Number")
    ax2.axvline(x=np.argmax(pdf_distances), color='r', linestyle='--', label='Detected Cut')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for main title
    plot_filename = f'{video_name}_comparison_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")
# --- REPLACE your old test block with this one ---
if __name__ == "__main__":
    # Define paths
    v1_path = "video1_category1.mp4"
    v2_path = "video2_category1.mp4"
    v3_path = "video3_category2.mp4"
    v4_path = "video4_category2.mp4"

    # --- Run the main function to get cut numbers ---
    print("--- Running Main Function ---")
    print(f"Cut for {v1_path}: {main(v1_path, 1)}")
    print(f"Cut for {v2_path}: {main(v2_path, 1)}")
    print(f"Cut for {v3_path}: {main(v3_path, 2)}")
    print(f"Cut for {v4_path}: {main(v4_path, 2)}")

    # --- Generate the plots for your report ---
    print("\n--- Generating Report Plots ---")

    # 1. Generate the standard distance plots for all 4 videos
    # (These are for your main results sections)
    generate_distance_plot(v1_path, "video1_category1")
    generate_distance_plot(v2_path, "video2_category1")
    generate_distance_plot(v3_path, "video3_category2")
    generate_distance_plot(v4_path, "video4_category2")

    # 2. Generate the special comparison plots
    # (These are your "proof" for the report)
    print("\n--- Generating Comparison Plots (Proof) ---")

    # --- ADD THIS LINE ---
    generate_comparison_plot(v1_path, "video1_category1")
    # ---------------------

    generate_comparison_plot(v2_path, "video2_category1")
    generate_comparison_plot(v3_path, "video3_category2")
    generate_comparison_plot(v4_path, "video4_category2")

    print("\nAll plots generated. You can now download the .png files.")

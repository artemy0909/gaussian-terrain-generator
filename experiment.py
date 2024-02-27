import datetime
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

RESEARCH_BASE_FOLDER = "results"
JSONS_FOLDER = "parameters"
RESULTS_FOLDER = "images"

height_limits = (-2 ** 14, 2 ** 14)


def generate_token(length: int):
    from random import choice
    from string import ascii_letters, digits
    return ''.join(choice(ascii_letters + digits) for _ in range(length))


def normalize_map(world, normalize=height_limits):
    from_value, to_value = normalize
    max_value = world.max()
    min_value = world.min()
    normalized_world = from_value + (world - min_value) * ((to_value - from_value) / (max_value - min_value))
    return normalized_world


def normalize_map_decorator(func):
    def wrapper(*args, **kwargs):
        world = func(*args, **kwargs)
        return normalize_map(world)
    return wrapper


def generate_world_egg(world_size):
    return np.random.uniform(*height_limits, size=(world_size, world_size))


def generate_mask(shape, mask_min=-0.2, mask_max=0.2, mask_smooth_iterations=1, mask_smooth_sigma=50):
    mask = np.random.uniform(mask_min, mask_max, size=shape)
    for _ in range(mask_smooth_iterations):
        mask = gaussian_filter(mask, sigma=mask_smooth_sigma)
    mask = normalize_map(mask, (mask_min, mask_max))
    # mask = np.abs(mask / np.max(mask))
    return mask


@normalize_map_decorator
def draw_world_texture(size):
    egg = generate_world_egg(size)
    texture = None
    for i in range(1, 5):
        mask = generate_mask((size, size), mask_smooth_sigma=10 * i)
        if texture is None:
            texture = gaussian_smooth_world(egg, 1, 25 / i) * mask
        else:
            texture += gaussian_smooth_world(egg, 1, 25 / i) * mask
    return texture


@normalize_map_decorator
def apply_world_texture(world_map):
    return world_map + draw_world_texture(world_map.shape[0])


@normalize_map_decorator
def gaussian_smooth_world(world, num_iterations, sigma):
    for i in range(num_iterations):
        world = gaussian_filter(world, sigma=sigma)
    return world


def easy_selectively_smooth(world, smoothing_factor):
    if smoothing_factor < 0 or smoothing_factor > 1:
        raise ValueError("Smoothing factor must be between 0 and 1.")
    mask = generate_mask(world.shape, mask_min=0, mask_max=1, mask_smooth_iterations=1, mask_smooth_sigma=50)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = 'terrain'
    im = ax.imshow(mask, cmap=cmap, interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Высота')
    plt.title("Маска")

    smoothed_world = world * (1 - smoothing_factor)
    smoothed_masked_world = world * (1 - mask) + smoothed_world * mask
    return smoothed_masked_world


@normalize_map_decorator
def add_noise(world, noise_factor):
    noise = np.random.uniform(-noise_factor, noise_factor, size=world.shape)
    return world + noise


@normalize_map_decorator
def sum_maps(*map_list):
    sum_ = None
    for map_ in map_list:
        if sum_ is None:
            sum_ = map_
        else:
            sum_ += map_
    return sum_


def gaussian_selectively_smooth(world, num_iterations, sigma):
    mask = generate_mask(world.shape)
    smoothed_world = world.copy()
    for i in range(num_iterations):
        smoothed_region = gaussian_filter(world, sigma=sigma)
        smoothed_world = smoothed_world * (1 - mask) + smoothed_region * mask
    return smoothed_world


def fill_by_land_area(world, land_area_percentage):
    sorted_world = np.sort(world, axis=None)
    land_index = int((1 - land_area_percentage / 100.0) * sorted_world.size)
    fill_level = sorted_world[land_index]
    filled_world = np.where(world <= fill_level, height_limits[0], world)
    return filled_world


def plot_world(world, title, params=None, save_path=None, classic_colors=True):
    def create_custom_colormap():
        colors = [(0, 0, 0.5), (0, 0, 0.7), (0, 0, 1), (0, 0.7, 1), (0, 0.5, 0), (0, 0.3, 0),
                  (0.5, 0.25, 0)]
        color_map = LinearSegmentedColormap.from_list('custom', colors, N=256)
        return color_map

    fig, ax = plt.subplots(figsize=(7, 5))
    if classic_colors:
        cmap = 'terrain'
    else:
        cmap = create_custom_colormap()
    im = ax.imshow(world, cmap=cmap, interpolation='bilinear', vmin=height_limits[0], vmax=height_limits[1])
    plt.colorbar(im, ax=ax, label='Высота')
    plt.title(title)

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= row < world.shape[0] and 0 <= col < world.shape[1]:
            z = world[row, col]
            return f'x={x:.2f}, y={y:.2f}, Высота={z:.2f}'
        else:
            return f'x={x:.2f}, y={y:.2f}'

    ax.format_coord = format_coord

    if params:
        param_str = '\n'.join([f'{key}: {value}' for key, value in params.items()])
        plt.annotate(param_str, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')


def save_parameters_to_json(params, save_path):
    with open(save_path, 'w') as json_file:
        json.dump(params, json_file, indent=4)


def land_counter(map, land_level):
    count = 0
    for x in map:
        for y in x:
            if y > land_level:
                count += 1
    return count


def create_continents(world, n_clusters=5):
    X, Y = np.indices(world.shape)
    points = np.column_stack((X.ravel(), Y.ravel(), world.ravel()))

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)

    labels = kmeans.labels_.reshape(world.shape)

    new_world = world.copy()
    for i in range(n_clusters):
        cluster_mean = world[labels == i].mean()
        new_world[labels == i] = cluster_mean

    return new_world

def start_research(**kwargs):
    global height_limits

    height_limits = kwargs.get('height_map_size', height_limits)
    world_size = kwargs.get('world_size', 100)
    gaussian_filter_num_iterations = kwargs.get('gaussian_filter_num_iterations', 10)
    noise_factor = kwargs.get('noise_factor', 100)
    sigma = kwargs.get('sigma', 1)
    seed = kwargs.get('seed', random.randint(0, 2 ** 32 - 1))
    show_plots = kwargs.get('show_plots', True)
    research_name = kwargs.get('research_name', 'unnamed')
    land_area = kwargs.get('fill_level', 30)
    generate_iterations = kwargs.get('generate_iterations', 1)

    np.random.seed(seed)

    current_research_folder_path = f"{RESEARCH_BASE_FOLDER}/{research_name}"
    current_results_folder_path = f"{current_research_folder_path}/{RESULTS_FOLDER}"
    current_jsons_folder_path = f"{current_research_folder_path}/{JSONS_FOLDER}"

    for path in [current_research_folder_path, current_results_folder_path, current_jsons_folder_path]:
        check_folder_exist(path)

    world_iterations = {}
    get_last_iter = lambda: list(world_iterations.values())[-1]

    smoothed_maps = []
    sigma_delta = sigma / generate_iterations
    filter_iter_delta = gaussian_filter_num_iterations / generate_iterations
    for i in range(generate_iterations):
        raw = generate_world_egg(world_size)
        smoothed_maps.append(gaussian_smooth_world(raw, round(filter_iter_delta * (i + 1)), sigma_delta * (i + 1)))
    world_iterations["sum"] = gaussian_smooth_world(sum_maps(*smoothed_maps), 10, 10)
    world_iterations["textured"] = apply_world_texture(get_last_iter())
    world_iterations["selectively_smoothed"] = gaussian_selectively_smooth(get_last_iter(), 1, 30)
    world_iterations["selectively_easy_smoothed"] = easy_selectively_smooth(get_last_iter(), 0.75)
    # world_iterations["noised"] = add_noise(get_last_iter(), noise_factor)
    # world_iterations["finaly_smoothed"] = smooth_world(get_last_iter(), 1, 2)

    # world_iterations["continented"] = create_continents(get_last_iter(), 10)

    if land_area:
        world_iterations["fiiled"] = fill_by_land_area(get_last_iter(), land_area)

    generate_number = generate_token(8)
    plot_path = f'{current_results_folder_path}/{datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")}' \
                f'_{generate_number}.png'
    # params = kwargs

    plot_world(get_last_iter(), f'{list(world_iterations.keys())[-1]} (last one)', params=kwargs, save_path=plot_path)
    json_path = f'{current_jsons_folder_path}/{datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")}' \
                f'_{generate_number}.json'
    save_parameters_to_json(kwargs, json_path)

    if show_plots:
        i = 1
        for key, value in world_iterations.items():
            # if value is not get_last_iter():
            plot_world(value, f'{key} (№ {i})')
            i += 1
        plt.show()


def check_folder_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


if __name__ == "__main__":
    from rich.progress import track

    for _ in track(range(100), description="Processing..."):
        start_research(
            world_size=1000,
            gaussian_filter_num_iterations=15,
            noise_factor=100,
            sigma=10,
            show_plots=True,
            research_name="test",
            generate_iterations=5,
            land_area=40,
        )

import random
import math
import colorsys
from typing import List, Dict, Tuple, Callable, Union, Optional, Any

# Constants
DEFAULT_SETTINGS = {
    "attempts": 1,
    "color_filter": None,
    "color_space": "default",
    "clustering": "k-means",
    "quality": 50,
    "ultra_precision": False,
    "distance": "euclidean",
    "seed": None
}

# Color space presets
PRESETS = {
    'all': [0, 360, 0, 100, 0, 100],
    'default': [0, 360, 30, 80, 35, 80],
    'sensible': [0, 360, 25.59, 55.59, 60.94, 90.94],
    'colorblind': [0, 360, 40, 70, 15, 85],
    'fancy-light': [0, 360, 15, 40, 70, 100],
    'fancy-dark': [0, 360, 8, 40, 7, 40],
    'shades': [0, 240, 0, 15, 0, 100],
    'tarnish': [0, 360, 0, 15, 30, 70],
    'pastel': [0, 360, 0, 30, 70, 100],
    'pimp': [0, 360, 30, 100, 25, 70],
    'intense': [0, 360, 20, 100, 15, 80],
    'fluo': [0, 300, 35, 100, 75, 100],
    'red-roses': [330, 20, 10, 100, 35, 100],
    'ochre-sand': [20, 60, 20, 50, 35, 100],
    'yellow-lime': [60, 90, 10, 100, 35, 100],
    'green-mint': [90, 150, 10, 100, 35, 100],
    'ice-cube': [150, 200, 0, 100, 35, 100],
    'blue-ocean': [220, 260, 8, 80, 0, 50],
    'indigo-night': [260, 290, 40, 100, 35, 100],
    'purple-wine': [290, 330, 0, 100, 0, 40]
}

VALID_CLUSTERINGS = {'force-vector', 'k-means'}
VALID_DISTANCES = {'euclidean', 'cmc', 'compromise', 'protanope', 'deuteranope', 'tritanope'}
VALID_PRESETS = set(PRESETS.keys())

# LAB color space constants
LAB_CONSTANTS = {
    'Kn': 18,
    'Xn': 0.95047,
    'Yn': 1,
    'Zn': 1.08883,
    't0': 0.137931034,  # 4 / 29
    't1': 0.206896552,  # 6 / 29
    't2': 0.12841855,   # 3 * t1 * t1
    't3': 0.008856452   # t1 * t1 * t1
}

# Color conversion functions
def xyz_to_rgb(r: float) -> int:
    return round(255 * (r <= 0.00304 and 12.92 * r or 1.055 * math.pow(r, 1 / 2.4) - 0.055))

def rgb_to_xyz_helper(r: float) -> float:
    r /= 255
    if r <= 0.04045:
        return r / 12.92
    return math.pow((r + 0.055) / 1.055, 2.4)

def xyz_to_lab(t: float) -> float:
    if t > LAB_CONSTANTS['t3']:
        return math.pow(t, 1 / 3)
    return t / LAB_CONSTANTS['t2'] + LAB_CONSTANTS['t0']

def rgb_to_xyz(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = rgb
    r = rgb_to_xyz_helper(r)
    g = rgb_to_xyz_helper(g)
    b = rgb_to_xyz_helper(b)

    x = xyz_to_lab((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / LAB_CONSTANTS['Xn'])
    y = xyz_to_lab((0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / LAB_CONSTANTS['Yn'])
    z = xyz_to_lab((0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / LAB_CONSTANTS['Zn'])

    return (x, y, z)

def lab_to_xyz(t: float) -> float:
    return t * t * t if t > LAB_CONSTANTS['t1'] else LAB_CONSTANTS['t2'] * (t - LAB_CONSTANTS['t0'])

def lab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    l, a, b = lab
    y = (l + 16) / 116
    x = y if math.isnan(a) else y + a / 500
    z = y if math.isnan(b) else y - b / 200

    y = LAB_CONSTANTS['Yn'] * lab_to_xyz(y)
    x = LAB_CONSTANTS['Xn'] * lab_to_xyz(x)
    z = LAB_CONSTANTS['Zn'] * lab_to_xyz(z)

    r = xyz_to_rgb(3.2404542 * x - 1.5371385 * y - 0.4985314 * z)
    g = xyz_to_rgb(-0.969266 * x + 1.8760108 * y + 0.041556 * z)
    b = xyz_to_rgb(0.0556434 * x - 0.2040259 * y + 1.0572252 * z)

    return (r, g, b)

def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    x, y, z = rgb_to_xyz(rgb)
    l = 116 * y - 16
    return (0 if l < 0 else l, 500 * (x - y), 200 * (y - z))

def validate_rgb(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255

def hex_pad(x: int) -> str:
    return ('0' + hex(x)[2:])[-2:]

def lab_to_rgb_hex(lab: Tuple[float, float, float]) -> str:
    rgb = lab_to_rgb(lab)
    return f"#{hex_pad(rgb[0])}{hex_pad(rgb[1])}{hex_pad(rgb[2])}"

def lab_to_hcl(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    l, a, b = lab
    c = math.sqrt(a * a + b * b)
    h = (math.atan2(b, a) * 180 / math.pi + 360) % 360
    
    if round(c * 10000) == 0:
        h = float('nan')
    
    return (h, c, l)

def diff_sort(distance: Callable, colors: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    colors = colors.copy()
    diff_colors = [colors.pop(0)]
    
    while colors:
        index = -1
        max_distance = float('-inf')
        
        for candidate_index, A in enumerate(colors):
            for B in diff_colors:
                d = distance(A, B)
                if d > max_distance:
                    max_distance = d
                    index = candidate_index
        
        diff_colors.append(colors[index])
        colors.pop(index)
    
    return diff_colors

def compute_quality_metrics(distance: Callable, colors: List[Tuple[float, float, float]]) -> Dict[str, float]:
    min_dist = float('inf')
    S = 0
    t = 0
    
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            d = distance(colors[i], colors[j])
            
            if d < min_dist:
                min_dist = d
            
            S += d
            t += 1
    
    return {"min": min_dist, "mean": S / t if t > 0 else 0}

# Distance functions
def euclidean_distance(lab1, lab2):
    dl = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return math.sqrt(dl * dl + da * da + db * db)

def distances_factory(distance_type="euclidean"):
    if distance_type == "euclidean":
        return euclidean_distance
    
    # Other distance functions could be implemented
    # For now just return euclidean
    return euclidean_distance

# String sum (for seed calculation)
def string_sum(s: str) -> int:
    sum_val = 0
    for char in s:
        sum_val += ord(char)
    return sum_val

def resolve_and_validate_settings(user_settings):
    settings = DEFAULT_SETTINGS.copy()
    if user_settings:
        settings.update(user_settings)
    
    if not isinstance(settings["attempts"], int) or settings["attempts"] <= 0:
        raise ValueError("iwanthue: invalid `attempts` setting. Expecting a positive number.")
    
    if settings["color_filter"] is not None and not callable(settings["color_filter"]):
        raise ValueError("iwanthue: invalid `color_filter` setting. Expecting a function.")
    
    if settings["clustering"] not in VALID_CLUSTERINGS:
        raise ValueError(f"iwanthue: unknown `clustering` '{settings['clustering']}'.")
    
    if not isinstance(settings["quality"], (int, float)) or math.isnan(settings["quality"]) or settings["quality"] < 1:
        raise ValueError("iwanthue: invalid `quality`. Expecting a number > 0.")
    
    if not isinstance(settings["ultra_precision"], bool):
        raise ValueError("iwanthue: invalid `ultra_precision`. Expecting a boolean.")
    
    if settings["distance"] not in VALID_DISTANCES:
        raise ValueError(f"iwanthue: unknown `distance` '{settings['distance']}'.")
    
    if isinstance(settings["seed"], str):
        settings["seed"] = string_sum(settings["seed"])
    
    if settings["seed"] is not None and not isinstance(settings["seed"], (int, float)):
        raise ValueError("iwanthue: invalid `seed`. Expecting an integer or a string.")
    
    # Building color filter from preset
    if not settings["color_filter"]:
        if settings["color_space"] and settings["color_space"] != "all":
            preset = None
            
            if isinstance(settings["color_space"], str):
                if settings["color_space"] not in VALID_PRESETS:
                    raise ValueError(f"iwanthue: unknown `color_space` '{settings['color_space']}'.")
                
                preset = PRESETS[settings["color_space"]]
            
            elif isinstance(settings["color_space"], list):
                if len(settings["color_space"]) != 6:
                    raise ValueError("iwanthue: expecting a `color_space` array of length 6 ([hmin, hmax, cmin, cmax, lmin, lmax]).")
                
                preset = settings["color_space"]
            
            else:
                cs = settings["color_space"]
                preset = [
                    cs.get("hmin", 0),
                    cs.get("hmax", 360),
                    cs.get("cmin", 0),
                    cs.get("cmax", 100),
                    cs.get("lmin", 0),
                    cs.get("lmax", 100)
                ]
            
            if preset[0] < preset[1]:
                def color_filter(rgb, lab):
                    hcl = lab_to_hcl(lab)
                    return (
                        hcl[0] >= preset[0] and hcl[0] <= preset[1] and
                        hcl[1] >= preset[2] and hcl[1] <= preset[3] and
                        hcl[2] >= preset[4] and hcl[2] <= preset[5]
                    )
                settings["color_filter"] = color_filter
            else:
                def color_filter(rgb, lab):
                    hcl = lab_to_hcl(lab)
                    return (
                        (hcl[0] >= preset[0] or hcl[0] <= preset[1]) and
                        hcl[1] >= preset[2] and hcl[1] <= preset[3] and
                        hcl[2] >= preset[4] and hcl[2] <= preset[5]
                    )
                settings["color_filter"] = color_filter
    
    return settings

def sample_lab_colors(rng, count, valid_color):
    colors = []
    
    for _ in range(count):
        while True:
            lab = [
                100 * rng(),
                100 * (2 * rng() - 1),
                100 * (2 * rng() - 1)
            ]
            
            rgb = lab_to_rgb(lab)
            
            if valid_color(rgb, lab):
                break
        
        colors.append(lab)
    
    return colors

# Constants for force vector algorithm
REPULSION = 100
SPEED = 100

def force_vector(rng, distance, valid_color, colors, settings):
    steps = settings["quality"] * 20
    
    while steps > 0:
        steps -= 1
        
        # Initialize vectors
        vectors = [{"dl": 0, "da": 0, "db": 0} for _ in range(len(colors))]
        
        # Compute forces
        for i in range(len(colors)):
            A = colors[i]
            
            for j in range(i):
                B = colors[j]
                
                # Repulsion
                d = distance(A, B)
                
                if d > 0:
                    dl = A[0] - B[0]
                    da = A[1] - B[1]
                    db = A[2] - B[2]
                    
                    force = REPULSION / (d * d)
                    
                    vectors[i]["dl"] += (dl * force) / d
                    vectors[i]["da"] += (da * force) / d
                    vectors[i]["db"] += (db * force) / d
                    
                    vectors[j]["dl"] -= (dl * force) / d
                    vectors[j]["da"] -= (da * force) / d
                    vectors[j]["db"] -= (db * force) / d
                else:
                    # Jitter
                    vectors[j]["dl"] += 2 - 4 * rng()
                    vectors[j]["da"] += 2 - 4 * rng()
                    vectors[j]["db"] += 2 - 4 * rng()
        
        # Apply forces
        for i in range(len(colors)):
            color = colors[i]
            displacement = SPEED * math.sqrt(
                vectors[i]["dl"] ** 2 +
                vectors[i]["da"] ** 2 +
                vectors[i]["db"] ** 2
            )
            
            if displacement > 0:
                ratio = (SPEED * min(0.1, displacement)) / displacement
                candidate_lab = [
                    color[0] + vectors[i]["dl"] * ratio,
                    color[1] + vectors[i]["da"] * ratio,
                    color[2] + vectors[i]["db"] * ratio
                ]
                
                rgb = lab_to_rgb(candidate_lab)
                
                if valid_color(rgb, candidate_lab):
                    colors[i] = candidate_lab

def k_means(distance, valid_color, colors, settings):
    color_samples = []
    
    # Amount of colors to try
    amount = settings["quality"] * 20
    
    if settings["ultra_precision"]:
        amount = amount * amount
    
    # Create a set of color samples
    rng_for_samples = random.Random(42)  # Fixed seed for consistency
    
    for _ in range(amount):
        lab = [
            100 * rng_for_samples.random(),
            100 * (2 * rng_for_samples.random() - 1),
            100 * (2 * rng_for_samples.random() - 1)
        ]
        
        rgb = lab_to_rgb(lab)
        
        if valid_color(rgb, lab):
            color_samples.append(lab)
    
    # K-means core
    if len(color_samples) < len(colors) * 10:
        # Not enough samples
        return
    
    # We need to initialize k-means
    samples_closest = [-1] * len(color_samples)
    
    # Iterate 20 times
    for _ in range(20):
        # Resetting centroids (mean color for each cluster)
        centroids_sum = [{"dl": 0, "da": 0, "db": 0, "count": 0} for _ in range(len(colors))]
        
        # Finding closest centroid for each sample
        for i, sample in enumerate(color_samples):
            min_dist = float('inf')
            closest_index = -1
            
            for j, centroid in enumerate(colors):
                dist = distance(sample, centroid)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_index = j
            
            if closest_index != samples_closest[i]:
                samples_closest[i] = closest_index
            
            # Adding to the centroid
            centroids_sum[closest_index]["dl"] += sample[0]
            centroids_sum[closest_index]["da"] += sample[1]
            centroids_sum[closest_index]["db"] += sample[2]
            centroids_sum[closest_index]["count"] += 1
        
        # Computing centroids
        for i in range(len(colors)):
            if centroids_sum[i]["count"] > 0:
                colors[i] = [
                    centroids_sum[i]["dl"] / centroids_sum[i]["count"],
                    centroids_sum[i]["da"] / centroids_sum[i]["count"],
                    centroids_sum[i]["db"] / centroids_sum[i]["count"]
                ]

def generate_palette(count, settings=None):
    # Validating arguments
    if not isinstance(count, int) or count < 1:
        raise ValueError("iwanthue: invalid `count`. Expecting a positive number.")
    
    # Empty palette
    if count == 0:
        return []
    
    # Only one color
    if count == 1:
        return ["#000000"]
    
    # Validation and resolution of settings
    settings = resolve_and_validate_settings(settings)
    
    # Distance function
    distance = distances_factory(settings["distance"])
    
    # Random
    seed = settings["seed"] if settings["seed"] is not None else random.randint(0, 1000000)
    rng = random.Random(seed)
    
    # Creating the valid color function
    def valid_color(rgb, lab):
        return (
            validate_rgb(rgb) and
            (settings["color_filter"] is None or settings["color_filter"](rgb, lab))
        )
    
    # Function used to sample colors
    def random_valid_colors():
        # Get lab colors
        lab_colors = sample_lab_colors(rng.random, count, valid_color)
        
        # Getting the relevant clustering method
        if settings["clustering"] == "k-means":
            k_means(distance, valid_color, lab_colors, settings)
        else:
            force_vector(rng.random, distance, valid_color, lab_colors, settings)
        
        # Sorting
        return diff_sort(distance, lab_colors)
    
    # Generating colors
    best_colors = None
    best_metrics = {"min": -1}
    
    for _ in range(settings["attempts"]):
        colors = random_valid_colors()
        
        if count < 2:
            best_colors = colors
            break
        
        metrics = compute_quality_metrics(distance, colors)
        
        if metrics["min"] > best_metrics["min"]:
            best_colors = colors
            best_metrics = metrics
    
    # Returning the hexadecimal colors
    return [lab_to_rgb_hex(lab) for lab in best_colors] 
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from matplotlib.widgets import Button, Slider, RadioButtons

# -----------------------
# 🌋 Volcano Info
# -----------------------
volcano_name = "Taal Volcano"
volcano_x, volcano_y = 2, -1

# -----------------------
# PHIVOLCS alert settings
# -----------------------
eruption_settings = {
    0: {"label": "Normal", "max_radius": 0, "growth_rate": 0,
        "colors": ['#ffe5cc', '#ffcc99'],
        "desc": "No eruption; normal background activity.",
        "eq_mag_str": "0.0", "eq_mag_num": 0.0},
    1: {"label": "Abnormal", "max_radius": 5, "growth_rate": 1,
        "colors": ['#ffb366', '#ff9933'],
        "desc": "Increased unrest; minor steam and gas emissions.",
        "eq_mag_str": "1.0–2.0", "eq_mag_num": 1.5},
    2: {"label": "Increasing Unrest", "max_radius": 12, "growth_rate": 1.5,
        "colors": ['#ff9933', '#ff8000'],
        "desc": "Possible minor eruptions; ashfall local.",
        "eq_mag_str": "2.0–3.5", "eq_mag_num": 3.0},
    3: {"label": "Minor Eruption", "max_radius": 25, "growth_rate": 2,
        "colors": ['#ff8000', '#ff6600'],
        "desc": "Phreatomagmatic eruption; widespread ashfall.",
        "eq_mag_str": "3.5–5.0", "eq_mag_num": 4.5},
    4: {"label": "Hazardous Eruption", "max_radius": 50, "growth_rate": 3,
        "colors": ['#ff6600', '#cc3300'],
        "desc": "Plinian-style eruption; heavy ashfall.",
        "eq_mag_str": "5.0+", "eq_mag_num": 6.0}
}

# -----------------------
# Map setup
# -----------------------
map_img = plt.imread("taal.png")  # make sure file exists
x_min, x_max, y_min, y_max = -60, 60, -60, 60

# Grid
grid_res = 300
xs = np.linspace(x_min, x_max, grid_res)
ys = np.linspace(y_min, y_max, grid_res)
XX, YY = np.meshgrid(xs, ys)
dist_grid = np.sqrt((XX - volcano_x) ** 2 + (YY - volcano_y) ** 2)

# -----------------------
# Damage and Ash Models
# -----------------------
def compute_damage_map(current_radius, scale, eq_mag_num, max_radius):
    with np.errstate(divide='ignore', invalid='ignore'):
        base = np.clip(1 - (dist_grid / current_radius), 0, 1) if current_radius > 0 else np.zeros_like(dist_grid)
    scale_factor = scale / 4.0
    quake_factor = min(eq_mag_num / 7.0, 1.0)
    damage = base * scale_factor * quake_factor
    if max_radius > 0:
        damage *= np.exp(-dist_grid / (max_radius / 4.0))
    damage = np.clip(damage, 0, 1)
    damage[dist_grid > max_radius] = 0.0
    return damage

def compute_ash_map(current_radius, wind_dir_deg, wind_speed_kmh, max_radius):
    ash_angle_deg = (wind_dir_deg + 180.0) % 360.0
    ash_rad = np.deg2rad(ash_angle_deg)
    ux, uy = math.sin(ash_rad), math.cos(ash_rad)
    wind_factor = max(0.1, wind_speed_kmh / 10.0)
    parallel_sigma = max(1.0, (current_radius + 1.0) * 0.4 * wind_factor)
    perp_sigma = max(0.5, (current_radius + 1.0) * 0.25)
    cx = volcano_x + ux * current_radius * 0.45
    cy = volcano_y + uy * current_radius * 0.45
    RX = XX - cx
    RY = YY - cy
    parallel = RX * ux + RY * uy
    perp = -RX * uy + RY * ux
    gauss = np.exp(-0.5 * ((parallel / parallel_sigma) ** 2 + (perp / perp_sigma) ** 2))
    gauss *= (1.0 / (1.0 + np.exp(-0.8 * parallel)))
    radial_atten = np.exp(-dist_grid / max(1.0, (max_radius / 3.0)))
    ash = gauss * radial_atten
    ash = ash / np.max(ash) if np.max(ash) > 0 else ash
    ash *= np.clip((current_radius / max(1.0, max_radius)) * 1.2 + 0.05, 0, 1)
    ash[dist_grid > max_radius * 1.5] = 0.0
    return np.clip(ash, 0, 1)

# -----------------------
# UI Setup
# -----------------------
plt.ion()
fig = plt.figure(figsize=(10, 9))
ax_map = fig.add_axes([0.05, 0.2, 0.7, 0.75])
cmap_damage, ash_cmap = plt.cm.Purples, plt.cm.Greys

# --- Controls layout ---
ax_alert = fig.add_axes([0.78, 0.80, 0.20, 0.14], facecolor='whitesmoke')
ax_wind_speed = fig.add_axes([0.78, 0.70, 0.18, 0.03], facecolor='lightgoldenrodyellow')
ax_wind_dir = fig.add_axes([0.78, 0.64, 0.18, 0.03], facecolor='lightgoldenrodyellow')
ax_ash_strength = fig.add_axes([0.78, 0.58, 0.18, 0.03], facecolor='lightgoldenrodyellow')
ax_btn_reset = fig.add_axes([0.78, 0.50, 0.18, 0.05])

radio_alert = RadioButtons(ax_alert, [f"{i}: {eruption_settings[i]['label']}" for i in range(5)], active=2)
slider_speed = Slider(ax_wind_speed, 'Wind Speed', 0, 50, valinit=10, valstep=1)
slider_dir = Slider(ax_wind_dir, 'Wind Dir', 0, 360, valinit=90, valstep=1)
slider_ash = Slider(ax_ash_strength, 'Ash Strength', 0.1, 2.0, valinit=1.0)
btn_reset = Button(ax_btn_reset, 'Reset Simulation', color='lightgray', hovercolor='0.9')

# State vars
show_ash = show_damage = show_impact = True
radius = 0.1
frame_dt = 0.5
scale = 2
settings = eruption_settings[scale]

# -----------------------
# Update Plot Function
# -----------------------
def update_plot():
    global radius, settings
    ax_map.clear()
    settings = eruption_settings[scale]
    max_radius, eq_mag_num = settings["max_radius"], settings["eq_mag_num"]

    ax_map.imshow(map_img, extent=[x_min, x_max, y_min, y_max], zorder=0)

    # Impact rings
    if show_impact:
        for i, c in enumerate(settings["colors"]):
            r = radius - i * 5
            if r > 0:
                ax_map.add_patch(plt.Circle((volcano_x, volcano_y), r, color=c, alpha=0.35, zorder=2))

    # Damage map
    if show_damage and radius > 0:
        dmg = compute_damage_map(radius, scale, eq_mag_num, max_radius)
        ax_map.imshow(np.ma.masked_where(dmg <= 0.001, dmg), extent=[x_min, x_max, y_min, y_max],
                      origin='lower', cmap=cmap_damage, alpha=0.6, vmin=0, vmax=1, zorder=3)

    # Ash map
    if show_ash and radius > 0:
        ash = compute_ash_map(radius, slider_dir.val, slider_speed.val, max_radius)
        ash = np.clip(ash * slider_ash.val, 0, 1)
        ax_map.imshow(np.ma.masked_where(ash <= 0.01, ash), extent=[x_min, x_max, y_min, y_max],
                      origin='lower', cmap=ash_cmap, alpha=0.6, vmin=0, vmax=1, zorder=4)

    # Volcano marker & info box
    ax_map.plot(volcano_x, volcano_y, 'r^', markersize=12, label=volcano_name, zorder=5)
    info = (f"Alert Level {scale} — {settings['label']}\n"
            f"{settings['desc']}\n"
            f"Wind {slider_speed.val:.1f} km/h from {slider_dir.val:.0f}°\n"
            f"Current Radius: {radius:.1f} / Max: {max_radius} km")
    ax_map.text(x_min + 2, y_max - 10, info, fontsize=9, color='black',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    ax_map.set_xlim(x_min, x_max)
    ax_map.set_ylim(y_min, y_max)
    ax_map.set_title(f"🌋 {volcano_name} — Alert Level {scale}: {settings['label']}", fontsize=12)
    ax_map.set_xlabel("Kilometers East-West")
    ax_map.set_ylabel("Kilometers North-South")
    fig.canvas.draw_idle()

# -----------------------
# UI Callbacks
# -----------------------
def on_alert_change(label):
    global scale, radius
    scale = int(label.split(":")[0])
    radius = 0.1
    update_plot()

def reset(event):
    global radius
    radius = 0.1
    update_plot()

radio_alert.on_clicked(on_alert_change)
slider_speed.on_changed(lambda val: update_plot())
slider_dir.on_changed(lambda val: update_plot())
slider_ash.on_changed(lambda val: update_plot())
btn_reset.on_clicked(reset)

# -----------------------
# Animation Loop
# -----------------------
plt.show(block=False)
while plt.fignum_exists(fig.number):
    settings = eruption_settings[scale]
    if radius <= settings["max_radius"]:
        radius += settings["growth_rate"] * 0.3
    update_plot()
    plt.pause(frame_dt)

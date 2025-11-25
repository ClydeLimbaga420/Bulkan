import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# ----------------------- Appearance defaults -----------------------
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ----------------------- Volcano Info -----------------------
volcano_name = "Taal Volcano"
volcano_x, volcano_y = 2, -1

# ----------------------- PHIVOLCS alert settings -----------------------
ERUPTION_COLORS = {
    0: {"label": "Normal", "max_radius": 0, "growth_rate": 0,
        "colors": ['#38761d', '#6aa84f'], "desc": "No eruption; normal background activity.",
        "eq_mag_str": "0.0", "eq_mag_num": 0.0},
    1: {"label": "Abnormal", "max_radius": 5, "growth_rate": 1,
        "colors": ['#f1c232', '#f6b26b'], "desc": "Increased unrest; minor steam and gas emissions.",
        "eq_mag_str": "1.0–2.0", "eq_mag_num": 1.5},
    2: {"label": "Increasing Unrest", "max_radius": 12, "growth_rate": 1.5,
        "colors": ['#cc0000', '#ff6600'], "desc": "Possible minor eruptions; ashfall local.",
        "eq_mag_str": "2.0–3.5", "eq_mag_num": 3.0},
    3: {"label": "Minor Eruption", "max_radius": 25, "growth_rate": 2,
        "colors": ['#800000', '#cc0000'], "desc": "Phreatomagmatic eruption; widespread ashfall.",
        "eq_mag_str": "3.5–5.0", "eq_mag_num": 4.5},
    4: {"label": "Hazardous Eruption", "max_radius": 50, "growth_rate": 3,
        "colors": ['#330000', '#800000'], "desc": "Plinian-style eruption; heavy ashfall.",
        "eq_mag_str": "5.0+", "eq_mag_num": 6.0}
}

# ----------------------- Map Setup -----------------------
try:
    map_img = plt.imread("taal.png")
except FileNotFoundError:
    print("Warning: 'taal.png' not found. Using blank background.")
    map_img = None

x_min, x_max, y_min, y_max = -60, 60, -60, 60
grid_res = 240
xs = np.linspace(x_min, x_max, grid_res)
ys = np.linspace(y_min, y_max, grid_res)
XX, YY = np.meshgrid(xs, ys)
dist_grid = np.sqrt((XX - volcano_x) ** 2 + (YY - volcano_y) ** 2)

# ----------------------- Damage & Ash Models -----------------------
def compute_damage_map(radius, scale, eq_mag_num, max_radius):
    if radius <= 0 or max_radius <= 0:
        return np.zeros_like(dist_grid)
    base = np.clip(1 - (dist_grid / radius), 0, 1)
    scale_factor = scale / 4.0
    quake_factor = min(eq_mag_num / 7.0, 1.0)
    damage = base * scale_factor * quake_factor
    damage *= np.exp(-dist_grid / max(1.0, (max_radius / 6.0)))
    damage[dist_grid > max_radius] = 0.0
    return np.clip(damage, 0, 1)

def compute_ash_map(radius, wind_dir, wind_speed, max_radius):
    if radius <= 0 or max_radius <= 0:
        return np.zeros_like(dist_grid)
    ash_angle_deg = (wind_dir + 180) % 360
    ash_rad = np.deg2rad(ash_angle_deg)
    ux, uy = math.sin(ash_rad), math.cos(ash_rad)
    wind_factor = max(0.1, wind_speed / 10.0)
    parallel_sigma = max(1.0, (radius + 1.0) * 0.4 * wind_factor)
    perp_sigma = max(0.5, (radius + 1.0) * 0.25)
    cx, cy = volcano_x + ux * radius * 0.45, volcano_y + uy * radius * 0.45
    RX, RY = XX - cx, YY - cy
    parallel = RX * ux + RY * uy
    perp = -RX * uy + RY * ux
    gauss = np.exp(-0.5 * ((parallel / parallel_sigma) ** 2 + (perp / perp_sigma) ** 2))
    gauss *= 1 / (1 + np.exp(-0.8 * parallel))
    radial_atten = np.exp(-dist_grid / max(1.0, max_radius / 3.0))
    ash = gauss * radial_atten
    ash /= np.max(ash) if np.max(ash) > 0 else 1
    ash *= np.clip((radius / max(1.0, max_radius)) * 1.2 + 0.05, 0, 1)
    ash[dist_grid > max_radius * 1.5] = 0
    return ash

# ----------------------- Figure & UI -----------------------
plt.ion()
fig = plt.figure(figsize=(12,10), facecolor='#ebebeb')
fig.suptitle("🌋 Taal Volcano Eruption Hazard Simulation", y=0.03, fontsize=16, fontweight='bold')
ax_map = fig.add_axes([0.05,0.12,0.7,0.82], facecolor='white')

colors = ["#fef0d9","#fc8d59","#d7301f"]
cmap_damage = LinearSegmentedColormap.from_list("DamageCmap", colors)
ash_cmap = plt.get_cmap('Greys_r')

# --- Controls ---
control_bg = '#f5f5f5'
ax_alert = fig.add_axes([0.78,0.80,0.2,0.15], facecolor=control_bg)
ax_wind_speed = fig.add_axes([0.78,0.73,0.18,0.035], facecolor=control_bg)
ax_wind_dir = fig.add_axes([0.78,0.68,0.18,0.035], facecolor=control_bg)
ax_ash_strength = fig.add_axes([0.78,0.63,0.18,0.035], facecolor=control_bg)
ax_btn_reset = fig.add_axes([0.78,0.56,0.18,0.045], facecolor=control_bg)
ax_btn_ash = fig.add_axes([0.78,0.49,0.18,0.045], facecolor=control_bg)
ax_btn_damage = fig.add_axes([0.78,0.42,0.18,0.045], facecolor=control_bg)
ax_btn_rings = fig.add_axes([0.78,0.35,0.18,0.045], facecolor=control_bg)

radio_alert = RadioButtons(ax_alert, [f"{i}: {ERUPTION_COLORS[i]['label']}" for i in range(5)], active=2)
radio_alert.ax.set_facecolor(control_bg)

slider_speed = Slider(ax_wind_speed, 'Wind Speed (km/h)', 0,50,valinit=10,valstep=1)
slider_dir = Slider(ax_wind_dir, 'Wind Dir (°)', 0,360,valinit=90,valstep=1)
slider_ash = Slider(ax_ash_strength,'Ash Scale',0.1,2.0,valinit=1.0)

btn_reset = Button(ax_btn_reset,'RESET SIMULATION',color='#dc143c',hovercolor='#ff6347')
btn_ash = Button(ax_btn_ash,'Ash Plume (ON)',color='#dcdcdc',hovercolor='#c0c0c0')
btn_damage = Button(ax_btn_damage,'Damage Map (ON)',color='#dcdcdc',hovercolor='#c0c0c0')
btn_rings = Button(ax_btn_rings,'Impact Rings (ON)',color='#dcdcdc',hovercolor='#c0c0c0')

# ----------------------- State -----------------------
show_ash, show_damage, show_impact = True, True, True
radius = 0.1
frame_dt = 0.45
scale = 2
settings = ERUPTION_COLORS[scale]

# ----------------------- Artists -----------------------
bg_img_artist = ax_map.imshow(map_img, extent=[x_min,x_max,y_min,y_max], zorder=0) if map_img is not None else None
dmg_artist = ax_map.imshow(np.zeros_like(XX), extent=[x_min,x_max,y_min,y_max], origin='lower', cmap=cmap_damage, alpha=0.85, vmin=0, vmax=1, zorder=3)
ash_artist = ax_map.imshow(np.zeros_like(XX), extent=[x_min,x_max,y_min,y_max], origin='lower', cmap=ash_cmap, alpha=0.65, vmin=0, vmax=1, zorder=4)
volcano_marker, = ax_map.plot(volcano_x,volcano_y,'^',markersize=16, markerfacecolor=settings["colors"][1], markeredgecolor='white', zorder=6)
info_text = ax_map.text(x_min+2, y_max-8, '', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.95, boxstyle='round,pad=0.6', linewidth=1.2), zorder=7)
impact_rings = [plt.Circle((volcano_x,volcano_y),1,color=settings["colors"][0], fill=False, linewidth=2.0, linestyle='--', alpha=0.55, zorder=2) for _ in settings["colors"]]
for ring in impact_rings: ax_map.add_patch(ring)

# ----------------------- Update Function -----------------------
def update_plot():
    global radius, settings
    settings = ERUPTION_COLORS[scale]
    max_radius, eq_mag_num = settings["max_radius"], settings["eq_mag_num"]

    # Damage
    dmg = compute_damage_map(radius, scale, eq_mag_num, max_radius) if show_damage else np.zeros_like(XX)
    dmg_artist.set_data(np.ma.masked_where(dmg <= 0.001, dmg))
    # Ash
    ash = compute_ash_map(radius, slider_dir.val, slider_speed.val, max_radius)*slider_ash.val if show_ash else np.zeros_like(XX)
    ash_artist.set_data(np.ma.masked_where(ash<=0.01,ash))
    # Volcano marker
    volcano_marker.set_markerfacecolor(settings["colors"][1])
    # Rings
    for i, ring in enumerate(impact_rings):
        r = radius - i*5
        ring.set_radius(r if show_impact and r>0 else 0)
        ring.set_edgecolor(settings["colors"][i])
    # Info
    info_text.set_text(
        f"Alert Level {scale}: {settings['label']}\n"
        f"Radius: {radius:.1f}/{max_radius} km\n"
        f"Wind: {slider_speed.val:.0f} km/h from {slider_dir.val:.0f}°\n"
        f"Possible Earthquake Magnitude: {settings['eq_mag_str']}\n"
        f"{settings['desc']}"
    )
    fig.canvas.draw_idle()
    # Update button labels ON/OFF
    btn_ash.label.set_text(f"Ash Plume ({'ON' if show_ash else 'OFF'})")
    btn_damage.label.set_text(f"Damage Map ({'ON' if show_damage else 'OFF'})")
    btn_rings.label.set_text(f"Impact Rings ({'ON' if show_impact else 'OFF'})")

# ----------------------- Callbacks -----------------------
def on_alert(label):
    global scale, radius
    scale = int(label.split(":")[0])
    radius = 0.1
    update_plot()
def reset(event):
    global radius, show_ash, show_damage, show_impact
    radius = 0.1
    show_ash = show_damage = show_impact = True
    slider_speed.set_val(10)
    slider_dir.set_val(90)
    slider_ash.set_val(1.0)
    update_plot()
def toggle_ash(event):
    global show_ash
    show_ash = not show_ash
    update_plot()
def toggle_damage(event):
    global show_damage
    show_damage = not show_damage
    update_plot()
def toggle_rings(event):
    global show_impact
    show_impact = not show_impact
    update_plot()

radio_alert.on_clicked(on_alert)
slider_speed.on_changed(lambda v: update_plot())
slider_dir.on_changed(lambda v: update_plot())
slider_ash.on_changed(lambda v: update_plot())
btn_reset.on_clicked(reset)
btn_ash.on_clicked(toggle_ash)
btn_damage.on_clicked(toggle_damage)
btn_rings.on_clicked(toggle_rings)

# ----------------------- Animation Loop -----------------------
update_plot()
plt.show(block=False)
while plt.fignum_exists(fig.number):
    settings = ERUPTION_COLORS[scale]
    if scale>0 and radius<settings["max_radius"]:
        radius += settings["growth_rate"]*0.3
        radius = min(radius, settings["max_radius"])
    elif scale==0 and radius>0.1:
        radius = max(0.1,radius-0.5)
    update_plot()
    plt.pause(frame_dt)

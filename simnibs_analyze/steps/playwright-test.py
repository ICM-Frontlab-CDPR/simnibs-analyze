import http.server, socketserver, threading, shutil, tempfile, json
from pathlib import Path
from contextlib import contextmanager



t1_path = "/home/hippolyted/Data/stimSD/_tmp_simnibs-outputs/1-simnibs-preps/0001/m2m_0001/T1.nii.gz"
efield_path = "/home/hippolyted/Data/stimSD/_tmp_simnibs-outputs/2-simnibs-simu/0001/simulations/simulation_AFFT-left_8dbde024/subject_volumes/0001_TDCS_1_scalar_magnE.nii.gz"
roi_path = "/home/hippolyted/Data/hemianotACS/data/derivatives/synthstroke-masks/0008/3DT1_4_lesion.nii.gz"



@contextmanager
def _serve(directory: Path):
    h = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(directory), **k)
    with socketserver.TCPServer(("127.0.0.1", 0), h) as httpd:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            yield httpd.server_address[1]
        finally:
            httpd.shutdown()


_HTML = """<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0"><canvas id="gl" width="{w}" height="{h}"></canvas>
<script type="module">
  import {{ Niivue, SLICE_TYPE }} from "{niivue_url}";
  const nv = new Niivue({nv_opts});
  window.nv = nv;
  await nv.attachTo("gl");
  await nv.loadVolumes({volumes});
  nv.setSliceType(SLICE_TYPE.RENDER);
  nv.setRenderAzimuthElevation({azim}, {elev});
  nv.drawScene();
  requestAnimationFrame(() => requestAnimationFrame(() => {{ window.__ready = true; }}));
</script></body></html>"""


def render_3d(
    volumes: list[dict],
    output_path: str | Path,
    azimuth: float = 120,
    elevation: float = 15,
    nv_opts: dict | None = None,
    width: int = 800,
    height: int = 600,
    niivue_url: str = "https://unpkg.com/@niivue/niivue/dist/index.js",
    timeout_ms: int = 60000,
) -> Path:
    """Rend n'importe quelle liste de volumes NiiVue en 3D, headless, vers un PNG.

    `volumes` : liste de dicts au format NiiVue, ex.
        [{"path": "T1.nii.gz", "colormap": "gray", "opacity": 1.0},
         {"path": "magnE.nii.gz", "colormap": "hot", "cal_min": 0.1,
          "cal_max": 1.0, "opacity": 0.7}]
    Chaque clé "path" est copiée dans un dossier servi localement et
    réécrite en "url" relative (NiiVue charge par fetch http, pas file://).
    """
    output_path = Path(output_path)
    nv_opts = nv_opts or {"backColor": [0.1, 0.1, 0.1, 1], "isColorbar": True}

    from playwright.sync_api import sync_playwright
    print('hello')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        served = []
        for i, vol in enumerate(volumes):
            v = dict(vol)
            src = v.pop("path", None)
            if src is None:
                raise ValueError(f"volume {i}: clé 'path' requise.")
            fname = f"vol{i}_{Path(src).name}"
            shutil.copy(src, tmp / fname)
            v["url"] = fname                 # url relative servie en http
            served.append(v)

        (tmp / "index.html").write_text(_HTML.format(
            w=width, h=height, niivue_url=niivue_url,
            nv_opts=json.dumps(nv_opts), volumes=json.dumps(served),
            azim=azimuth, elev=elevation,
        ))

        with _serve(tmp) as port:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--use-gl=angle", "--use-angle=swiftshader"],  # WebGL sans GPU
                )
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(f"http://127.0.0.1:{port}/index.html")
                page.wait_for_function("window.__ready === true", timeout=timeout_ms)
                with page.expect_download() as dl:
                    page.evaluate("nv.saveScene('scene.png')")
                dl.value.save_as(str(output_path))
                browser.close()
    return output_path


# vols=[
# {"path": t1_path,     "colormap": "gray", "opacity": 1.0},
# {"path": efield_path, "colormap": "hot",  "cal_min": 0.1,
# "cal_max": 1.0, "opacity": 0.7},
# {"path": roi_path,    "colormap": "red",  "opacity": 0.4},
# ]


# render_3d(
#     volumes=vols,
#     output_path="efield_sub-02-test.png",
#     azimuth=120, elevation=15,
# )


# #some examples
# render_3d(volumes=vols, output_path="vue_gauche.png",  azimuth=270, elevation=0)   # profil gauche
# render_3d(volumes=vols, output_path="vue_droite.png",  azimuth=90,  elevation=0)   # profil droit
# render_3d(volumes=vols, output_path="vue_face.png",    azimuth=180, elevation=0)   # face (antérieur)
# render_3d(volumes=vols, output_path="vue_dessus.png",  azimuth=0,   elevation=90)  # vue du dessus
# render_3d(volumes=vols, output_path="vue_3qrt.png",    azimuth=135, elevation=20)  # trois-quarts



# for az in range(0, 360, 45):
#     render_3d(volumes=vols, output_path=f"rot_{az:03d}.png", azimuth=az, elevation=15)
    
    
# vols = [
#     {"path": t1_path,     "colormap": "gray",    "opacity": 1.0},
#     {"path": efield_path, "colormap": "viridis", "cal_min": 0.1, "cal_max": 1.0, "opacity": 0.8},
# ]

# render_3d(volumes=vols, output_path="vue_gauche.png",  azimuth=270, elevation=0)



# ====================== EXEMPLES ======================

vols = [
    {"path": t1_path,     "colormap": "gray", "opacity": 1.0},
    {"path": efield_path, "colormap": "hot",  "cal_min": 0.1, "cal_max": 1.0, "opacity": 0.7},
    {"path": roi_path,    "colormap": "red",  "opacity": 0.4},
]

# Rendu de base
render_3d(vols, "efield_sub-02-test.png", azimuth=120, elevation=15)

# Orientations (caméra)
render_3d(vols, "vue_gauche.png", azimuth=270, elevation=0)    # profil gauche
render_3d(vols, "vue_droite.png", azimuth=90,  elevation=0)    # profil droit
render_3d(vols, "vue_face.png",   azimuth=180, elevation=0)    # face (antérieur)
render_3d(vols, "vue_dessus.png", azimuth=0,   elevation=90)   # vue du dessus
render_3d(vols, "vue_3qrt.png",   azimuth=135, elevation=20)   # trois-quarts

# Balayage multi-angles (pour un GIF / une planche)
for az in range(0, 360, 45):
    render_3d(vols, f"rot_{az:03d}.png", azimuth=az, elevation=15)

# Colormap différente (viridis) — T1 + E-field seulement
vols_viridis = [
    {"path": t1_path,     "colormap": "gray",    "opacity": 1.0},
    {"path": efield_path, "colormap": "viridis", "cal_min": 0.1, "cal_max": 1.0, "opacity": 0.8},
]
render_3d(vols_viridis, "vue_viridis.png", azimuth=270, elevation=0)

# Opacité : T1 translucide pour voir le champ "à travers"
vols_transp = [
    {"path": t1_path,     "colormap": "gray", "opacity": 0.5},
    {"path": efield_path, "colormap": "hot",  "cal_min": 0.1, "cal_max": 1.0, "opacity": 0.9},
    {"path": roi_path,    "colormap": "red",  "opacity": 0.3},
]
render_3d(vols_transp, "vue_opacite.png", azimuth=135, elevation=20)

# Seuillage : ne garder que les valeurs fortes du champ
vols_seuil = [
    {"path": t1_path,     "colormap": "gray", "opacity": 1.0},
    {"path": efield_path, "colormap": "hot",  "cal_min": 0.3, "cal_max": 0.8, "opacity": 0.9},
]
render_3d(vols_seuil, "vue_seuil.png", azimuth=135, elevation=20)

# Options de scène : fond blanc + cube d'orientation (figure de publication)
opts_pub = {"backColor": [1, 1, 1, 1], "isColorbar": True, "isOrientCube": True, "isRuler": False}
render_3d(vols, "vue_publication.png", nv_opts=opts_pub, azimuth=135, elevation=20)
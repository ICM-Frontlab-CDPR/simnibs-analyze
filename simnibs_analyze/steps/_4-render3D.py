import http.server
import socketserver
import threading
import shutil
import tempfile
import json
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def _serve(directory: Path):
    h = lambda *a, **k: http.server.SimpleHTTPRequestHandler(
        *a, directory=str(directory), **k
    )
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

    print("hello")
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
            v["url"] = fname  # url relative servie en http
            served.append(v)

        (tmp / "index.html").write_text(
            _HTML.format(
                w=width,
                h=height,
                niivue_url=niivue_url,
                nv_opts=json.dumps(nv_opts),
                volumes=json.dumps(served),
                azim=azimuth,
                elev=elevation,
            )
        )

        with _serve(tmp) as port:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--use-gl=angle",
                        "--use-angle=swiftshader",
                    ],  # WebGL sans GPU
                )
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(f"http://127.0.0.1:{port}/index.html")
                page.wait_for_function("window.__ready === true", timeout=timeout_ms)
                with page.expect_download() as dl:
                    page.evaluate("nv.saveScene('scene.png')")
                dl.value.save_as(str(output_path))
                browser.close()
    return output_path

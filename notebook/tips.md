---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import synthnf.inspection as ins
import synthnf.inspection.swing_supressing_inspection as ssi

from tqdm.auto import tqdm
import logging
mi.set_log_level(mi.LogLevel.Error)

import synthnf.simulation as simu
import synthnf.inspection as inspection

import numpy as np
import matplotlib.pyplot as plt

import pathlib
import tempfile

import synthnf.io as io
import synthnf.config.defaults as defaults
import synthnf.config.assets as assets
import synthnf.scene as scene
```

```python

from synthnf.inspection import FAInspection


class ControlledInspection(FAInspection):
    def __init__(self, params):
        super(self.__class__, self).__init__(params)
        
    def render_frame(
            self, 
            cam_z, 
            light_intensity,
            cam_fov = 28,
            face_num=1, 
            has_alpha = False
    ):
        scene_dict = scene.inspection_dict(
            cam_z=cam_z, 
            face_num=face_num,
            cam_ligth_intensity=light_intensity,
            cam_fov = cam_fov
        )

        scene_dict["tips"] = self.model_tips.copy()
        scene_dict["butts"] = self.model_butts.copy()
        scene_dict["model"] = self.model_fa.copy()

        keys = [f"model_grid_{int(i)}" for i in range(len(self.model_grids))]
        for key, g in zip(keys, self.model_grids):
            scene_dict[key] = g.copy()

        model_keys = ["tips", "butts", "model", *keys]
        if has_alpha:
            return scene.render_scene(scene_dict, spp=128*8, alpha=True, denoise=False)
        else:
            return scene.render_scene(scene_dict, spp=128, alpha=False, denoise=True)
    
def render_tips(
    light_intensity = 50,
    cloudiness = .7,
    y = 3800,
    z_displacement = 0,
    cam_fov = 28,
    has_alpha = False,
    seed = 123,
):
    params = simu.RandomParameters(
        seed = seed,
        cloudiness=cloudiness,
        max_z_displacement_mm=z_displacement,
        n_textures = 11
    )
    fains = ControlledInspection(
        params,
    )
    return fains.render_frame(
        y,
        light_intensity = light_intensity,
        has_alpha=has_alpha,
        cam_fov = cam_fov
    )


```

```python
base_frame = render_tips()
plt.imshow(base_frame)
base_frame.shape
```

# Jas

Lze naimpolementovat na úrovni obrazu nebo na renderigu. V rámci renderování je přidáme na intenzitě světla

```python
burnt = render_tips(light_intensity=200)
dim = render_tips(light_intensity=10)

fig,(axr,axc, axl) = plt.subplots(1,3,figsize=(20,10))
axr.imshow(base_frame)
axc.imshow(burnt)
axl.imshow(dim)
```

# Mat/Odrazivost

Matnost povrchu je úměrná oxidaci. Matné je to úměrně k hodnotě `cloudiness`

```python
refl = render_tips(cloudiness=.4)
matte = render_tips(cloudiness=.8)

fig,(axr,axc, axl) = plt.subplots(1,3,figsize=(20,10))
axr.imshow(base_frame)
axc.imshow(refl)
axl.imshow(matte)
```

# Obraz

- proutky se vystrkují pomocí `z displacemnet` parametru. jedná se o maximální displacement. Ten reálný je generovaný náhodně (náhodu lze kontrolovat pomocí parametru `seed`)
- zaměření kamery lze měnit pomocí `fov`
- posun v ose y  - `y` (defaultní hodnota je 3800, protože to je výška ve které se špičky nachází  a 0 je patice)

```python
shifted_y = render_tips(y = 3825)
plt.title("y shifted")
plt.imshow(np.hstack([base_frame, shifted_y]))
plt.show()

plt.title("displacement")
displaced_123 = render_tips(z_displacement=10,seed = 123)
displaced_321 = render_tips(z_displacement=10,seed = 321)
plt.imshow(np.vstack([displaced_123[150:350],displaced_321[150:350]]))
plt.show()


wide = render_tips(cam_fov = 45)
narrow = render_tips(cam_fov = 24)

_,axs=plt.subplots(1,3,figsize=(30,10))
plt.suptitle("fov")
axs[0].imshow(base_frame)
axs[1].imshow(wide)
axs[2].imshow(narrow)
plt.show()


```

# Fleky

Vygeneruj si fleky. Pak je buď 

1) nalep na obrázek
2) dej na pozadí (viz kód níže).

Poznámka: K tomu, aby šly špičky vygenerovat jen jako popředí je nutné změnit rendering proceduru. Pro tebe to znamená, že se bude obrázek renderovat **výrazně** déle a bude lehce zašuměný

```python
import io
import requests
from PIL import Image
import matplotlib.pyplot as plt  
import scipy.ndimage as ndi

url = 'https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1000w,f_auto,q_auto:best/msnbc/Components/Photos/040408/040408_ancientcat_cat.jpg'
data = requests.get(url).content
background = np.array(Image.open(io.BytesIO(data)))
h,w = base_frame.shape[:2]

foreground = render_tips(has_alpha=True)
plt.imshow(background[:h,:w])
plt.imshow(foreground)
```

# Blur / Šum

Naimpolementuj na urovni obrazu e.g. 

```python
import scipy.ndimage as ndi

plt.imshow(ndi.gaussian_filter(base_frame,3))
plt.show()
# kdyby byla potřeba implementovat ostrou rovinu, je nutné zkusit toto:
# https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html#perspective-camera-with-a-thin-lens-thinlens
```

# Model špičky a gridu

Model špiček lze změnit. Otevři si `assets/tip_model.ply` v [Blenderu](https://www.blender.org/) a model si předělej. To samé platí pro `assets/grid_wta.ply` (tam najdeš jen jeden zub, který se pak v runtime rozkopíruje).

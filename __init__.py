bl_info = {
    "name": "Speech To Blender",
    "author": "STB",
    "version": (0, 5, 0),
    "blender": (3, 6, 0),
    "category": "System",
    "description": "Voice Tools for Blender",
}

import bpy
from bpy.props import StringProperty, EnumProperty
from bpy.types import AddonPreferences, Operator, Panel

ADDON_ROOT = (__package__ or __name__).split(".")[0]  # "SpeechToBlender"

# ───────── Preferences (single class) ─────────
class STB_AddonPreferences(AddonPreferences):
    bl_idname = ADDON_ROOT

    meshy_api_key: StringProperty(
        name="Meshy API Key",
        description="Stored locally in Blender prefs",
        subtype="PASSWORD",
        default="",
    )
    meshy_base_url: StringProperty(
        name="Meshy Base URL",
        default="https://api.meshy.ai",
    )
    meshy_model: StringProperty(
        name="Meshy Model",
        description="e.g. v5",
        default="v5",
    )
    meshy_mode: EnumProperty(
        name="Mode",
        items=[
            ("preview", "Preview", "Faster, lower cost"),
            ("standard", "Standard", "Full quality"),
        ],
        default="standard",
    )
    meshy_formats: StringProperty(
        name="Export Formats",
        description="Comma separated: glb,fbx,obj",
        default="glb,fbx,obj",
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.prop(self, "meshy_api_key")
        col.prop(self, "meshy_base_url")
        col.prop(self, "meshy_model")
        col.prop(self, "meshy_mode")
        col.prop(self, "meshy_formats")


# ───────── Minimal operator and panels so UI never crashes ─────────
class STB_OT_MeshyGenerate(bpy.types.Operator):
    bl_idname = "stb.meshy_generate"
    bl_label = "Generate with Meshy"

    def execute(self, context):
        try:
            from .stb_core.providers import meshy as meshy_provider
        except Exception as e:
            self.report({'ERROR'}, f"core import failed: {e}")
            return {'CANCELLED'}

        prompt = context.window_manager.stb_meshy_prompt or "simple low-poly test object"

        # 👇 get prefs safely
        try:
            addon_prefs = context.preferences.addons[ADDON_ROOT].preferences
        except Exception:
            addon_prefs = None

        cfg = {
            "api_key": getattr(addon_prefs, "meshy_api_key", "") or os.environ.get("MESHY_API_KEY", ""),
            "base_url": getattr(addon_prefs, "meshy_base_url", "https://api.meshy.ai"),
            "model":    getattr(addon_prefs, "meshy_model", "v5"),
            "mode":     getattr(addon_prefs, "meshy_mode", "standard"),
            "formats":  getattr(addon_prefs, "meshy_formats", "glb,fbx,obj"),
        }

       

        try:
            job = meshy_provider.generate_from_prompt(context, prompt, cfg=cfg)
            self.report({'INFO'}, f"Meshy job: {job}")
        except Exception as e:
            self.report({'ERROR'}, f"Meshy call failed: {e}")
            return {'CANCELLED'}
        
        res = meshy_provider.generate_from_prompt(context, prompt, cfg=cfg)
        self.report({'INFO'}, f"Meshy job submitted: {res}")

        return {'FINISHED'}


class STB_PT_MeshyTools(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STB"
    bl_label = "Meshy Tools"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        layout.prop(wm, "stb_meshy_prompt", text="Prompt")
        layout.operator("stb.meshy_generate", icon="MESH_CUBE", text="Generate")


class STB_PT_MeshyStatus(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "STB"
    bl_label = "Meshy Status"
    bl_parent_id = ""
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        try:
            from .stb_core.providers import meshy as meshy_provider
            status = meshy_provider.get_meshy_status()
        except Exception as e:
            status = f"core not ready: {e}"
        layout.label(text=f"Status: {status}")


# ───────── Safe stub: lazy import inside register, timers last ─────────
_CLASSES = (
    STB_AddonPreferences,
    STB_OT_MeshyGenerate,
    STB_PT_MeshyTools,
    STB_PT_MeshyStatus,
)

def register():
    # 1) prefs first
    bpy.utils.register_class(STB_AddonPreferences)

    # 2) ensure WM prop before any panel draw
    if not hasattr(bpy.types.WindowManager, "stb_meshy_prompt"):
        bpy.types.WindowManager.stb_meshy_prompt = StringProperty(
            name="Meshy Prompt",
            default="simple low-poly test object",
        )

    # 3) operators and panels
    bpy.utils.register_class(STB_OT_MeshyGenerate)
    bpy.utils.register_class(STB_PT_MeshyTools)
    bpy.utils.register_class(STB_PT_MeshyStatus)

    # 4) lazy import real module, never crash on error
    try:
        # Import your heavy modules late
        from . import stb_core  # noqa: F401
    except Exception as e:
        # do not raise, just log so Blender does not auto‑disable
        print("[SpeechToBlender] STARTUP ERROR:", e)

    # 5) timers last if you add them later


def unregister():
    # panels and ops
    for cls in reversed(_CLASSES[1:]):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    # prefs
    try:
        bpy.utils.unregister_class(STB_AddonPreferences)
    except Exception:
        pass

    # WM prop
    try:
        if hasattr(bpy.types.WindowManager, "stb_meshy_prompt"):
            del bpy.types.WindowManager.stb_meshy_prompt
    except Exception:
        pass
